"""Frozen multilingual E5 teacher, cached on TRAIN only."""
from pathlib import Path
import hashlib
import json
import torch
from torch.nn import functional as F

MODEL="intfloat/multilingual-e5-small"
REVISION="fd1525a9fd15316a2d503bf26ab031a61d056e98"

def encode(model,tokenizer,texts,device,batch_size=16):
    result=[]
    # Sort by length to reduce padding, then restore order.
    order=sorted(range(len(texts)),key=lambda i:len(texts[i]))
    out=torch.empty((len(texts),model.config.hidden_size))
    with torch.inference_mode():
        for start in range(0,len(order),batch_size):
            ids=order[start:start+batch_size]
            batch=tokenizer([texts[i] for i in ids],padding=True,truncation=True,max_length=512,return_tensors="pt")
            batch={k:v.to(device) for k,v in batch.items()}
            hidden=model(**batch).last_hidden_state
            mask=batch["attention_mask"].unsqueeze(-1)
            vectors=F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
            out[ids]=vectors.cpu()
            if start%512==0:
                print(f"E5 encoded {start+len(ids)}/{len(order)}",flush=True)
    return out

class SemanticTeacher:
    def __init__(self,rows,examples,device="cpu",cache_dir=None):
        self.doc_ids={r["id"]:i for i,r in enumerate(rows)}
        self.query_ids={e["id"]:i for i,e in enumerate(examples)}
        dataset_hash=hashlib.sha256(json.dumps({
            "documents":[(r["id"],r["encoder_text"]) for r in rows],
            "queries":[(e["id"],e["query"]) for e in examples],
        },ensure_ascii=False,separators=(",",":")).encode("utf-8")).hexdigest()
        metadata={"model":MODEL,"revision":REVISION,"dataset_sha256":dataset_hash,
                  "document_ids":list(self.doc_ids),"query_ids":list(self.query_ids)}
        cache_path=Path(cache_dir)/"e5-train-embeddings.pt" if cache_dir else None
        manifest_path=cache_path.with_suffix(".manifest.json") if cache_path else None
        if cache_path and cache_path.exists() and manifest_path.exists():
            cache_manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
            cache_hash=hashlib.sha256(cache_path.read_bytes()).hexdigest()
            cached=torch.load(cache_path,map_location="cpu",weights_only=True)
            if (cache_hash==cache_manifest.get("cache_sha256") and cache_manifest.get("metadata")==metadata
                    and cached.get("metadata")==metadata and cached["docs"].shape[0]==len(rows)
                    and cached["queries"].shape[0]==len(examples)):
                self.docs=cached["docs"].float(); self.queries=cached["queries"].float()
                print(f"E5 cache hit: {cache_path}",flush=True)
                return
        from transformers import AutoModel,AutoTokenizer
        tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
        model=AutoModel.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,use_safetensors=True).to(device).eval()
        self.docs=encode(model,tok,["passage: "+r["encoder_text"] for r in rows],device)
        self.queries=encode(model,tok,["query: "+e["query"] for e in examples],device)
        del model,tok
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        if cache_path:
            cache_path.parent.mkdir(parents=True,exist_ok=True)
            tmp=cache_path.with_suffix(cache_path.suffix+".tmp")
            torch.save({"metadata":metadata,"docs":self.docs,"queries":self.queries},tmp)
            tmp.replace(cache_path)
            manifest_tmp=manifest_path.with_suffix(manifest_path.suffix+".tmp")
            manifest_tmp.write_text(json.dumps({"metadata":metadata,
                "cache_sha256":hashlib.sha256(cache_path.read_bytes()).hexdigest()},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            manifest_tmp.replace(manifest_path)

    def scores(self,examples,ids,device):
        q=self.queries[[self.query_ids[e["id"]] for e in examples]]
        d=self.docs[[self.doc_ids[k] for k in ids]]
        return (q@d.T).to(device)

    def projected_vectors(self, dimensions, device):
        """Use the same train-only orthogonal projection for query and passage vectors."""
        vectors=torch.cat((self.queries,self.docs)).to(device)
        _,eigenvectors=torch.linalg.eigh(vectors.T @ vectors)
        projected=F.normalize(vectors @ eigenvectors[:, -dimensions:],dim=-1)
        return projected.split((len(self.queries),len(self.docs)))

def semantic_loss(student_logits,teacher_cosine,positive,weights,teacher_temperature=.025):
    # Distil the relative ranking only among selected candidates. Ignore
    # possible equivalents in the same way as the contrastive objective.
    keep=(weights>0)|positive
    t=(teacher_cosine/teacher_temperature).masked_fill(~keep,-1e4)
    s=student_logits.masked_fill(~keep,-1e4)
    # Very confident teacher matches among nominal negatives are not
    # forced apart by InfoNCE; caller adjusts their negative weight.
    return F.kl_div(F.log_softmax(s,dim=1),F.softmax(t,dim=1),reduction="batchmean")
