"""Frozen multilingual E5 teacher, cached on TRAIN only."""
from pathlib import Path
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
    def __init__(self,rows,examples,device="cpu"):
        from transformers import AutoModel,AutoTokenizer
        tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
        model=AutoModel.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,use_safetensors=True).to(device).eval()
        self.doc_ids={r["id"]:i for i,r in enumerate(rows)}
        self.query_ids={e["id"]:i for i,e in enumerate(examples)}
        self.docs=encode(model,tok,["passage: "+r["encoder_text"] for r in rows],device)
        self.queries=encode(model,tok,["query: "+e["query"] for e in examples],device)
        del model,tok
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    def scores(self,examples,ids,device):
        q=self.queries[[self.query_ids[e["id"]] for e in examples]]
        d=self.docs[[self.doc_ids[k] for k in ids]]
        return (q@d.T).to(device)

def semantic_loss(student_logits,teacher_cosine,positive,weights,teacher_temperature=.025):
    # Distil the relative ranking only among selected candidates. Ignore
    # possible equivalents in the same way as the contrastive objective.
    keep=(weights>0)|positive
    t=(teacher_cosine/teacher_temperature).masked_fill(~keep,-1e4)
    s=student_logits.masked_fill(~keep,-1e4)
    # Very confident teacher matches among nominal negatives are not
    # forced apart by InfoNCE; caller adjusts their negative weight.
    return F.kl_div(F.log_softmax(s,dim=1),F.softmax(t,dim=1),reduction="batchmean")
