"""Train University Encoder from scratch, with a weak BM25F teacher."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter

import torch

from ai.indexing.chunker import digest
from ai.model.checkpoint import save_checkpoint, load_checkpoint, tokenizer_sha256
from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder
from ai.model.inputs import collate_texts
from ai.search.bm25 import BM25
from ai.search.lexical import tokens
from ai.tokenizer.bpe import BPETokenizer
from ai.training.dataset import VERSION, SPLITS, validate, normalize, near_duplicate
from ai.training.losses import contrastive_loss, teacher_loss
from ai.training.retrieval_eval import evaluate, choose_alpha

def read_dataset(directory,tokenizer_path,tokenizer):
    directory=Path(directory)
    manifest=json.loads((directory/"dataset-manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"]!=VERSION or manifest["tokenizer_sha256"]!=tokenizer_sha256(tokenizer_path):
        raise ValueError("dataset/tokenizer mismatch")
    data={}
    for split in SPLITS:
        parts=[]
        for kind in ("passages","examples"):
            name=f"{split}-{kind}.jsonl"
            raw=(directory/name).read_bytes().decode("utf-8")
            if digest(raw)!=manifest["files"][name]:
                raise ValueError("dataset integrity mismatch: "+name)
            rows=[json.loads(x) for x in raw.splitlines() if x]
            if len(rows)!=manifest["splits"][split][kind]:
                raise ValueError("dataset count mismatch")
            if any(r["split"]!=split for r in rows):
                raise ValueError("dataset split mismatch")
            parts.append(rows)
        data[split]=tuple(parts)
    validate(data,tokenizer,manifest["max_length"])
    return data,manifest

class TrainingCorpus:
    def __init__(self,rows,examples,tokenizer):
        self.rows={r["id"]:r for r in rows}
        self.examples=examples
        self.terms={k:set(tokens(r["text"])) for k,r in self.rows.items()}
        self.query_terms={e["id"]:set(tokens(e["query"])) for e in examples}
        self.lengths={k:len(tokenizer.encode(r["encoder_text"])) for k,r in self.rows.items()}
        self.lex=BM25(rows)
        self.teacher_cache={}
        self.headings={k:normalize(r["headings"][-1]) if r["headings"] else "" for k,r in self.rows.items()}
        self.bodies={k:normalize(r["text"]) for k,r in self.rows.items()}

    def teacher(self,e):
        key=e["id"]
        if key not in self.teacher_cache:
            self.teacher_cache[key]={h.passage["id"]:h.score for h in self.lex.search(
                e["query"],cohort=e["cohort"],subject=e["subject"],limit=60,fuzzy=False)}
        return self.teacher_cache[key]

    def batches(self,rng,batch_size,max_steps):
        order=list(self.examples)
        rng.shuffle(order)
        # Sort only small shuffled pools by approximate length: reduce pad
        # cost without systematically training on short passages first.
        batches=[]
        width=batch_size*12
        for start in range(0,len(order),width):
            block=sorted(order[start:start+width],key=lambda e:max(self.lengths[k] for k in e["positive_ids"]))
            batches.extend(block[i:i+batch_size] for i in range(0,len(block),batch_size))
        rng.shuffle(batches)
        return batches[:max_steps] if max_steps else batches

    def prepare(self,examples,rng,device):
        chosen=[rng.choice(e["positive_ids"]) for e in examples]
        hard={}
        for i,e in enumerate(examples):
            if i%3==0 and e["negative_candidates"]:
                candidate=rng.choice(e["negative_candidates"])["id"]
                # The lexical teacher can identify uncertain hard labels.
                hard[i]=candidate
        ids=list(dict.fromkeys(chosen+list(hard.values())))
        index={k:i for i,k in enumerate(ids)}
        positive=torch.zeros((len(examples),len(ids)),dtype=torch.bool)
        weights=torch.ones((len(examples),len(ids)))
        teacher_scores=torch.zeros_like(weights)
        teacher_pairs=[]
        for i,e in enumerate(examples):
            pos=set(e["positive_ids"])
            p_rows=[self.rows[k] for k in pos]
            p_sources={r["source"] for r in p_rows}
            p_heads={self.headings[k] for k in pos}-{""}
            p_bodies={self.bodies[k] for k in pos}
            candidates={c["id"] for c in e["negative_candidates"]}
            for j,k in enumerate(ids):
                if k in pos:
                    positive[i,j]=True
                    continue
                r=self.rows[k]
                # Ignore potentially equivalent passages, including different
                # known positives, same source, heading and near duplicates.
                if (r["source"] in p_sources or self.headings[k] in p_heads
                    or self.bodies[k] in p_bodies
                    or any(near_duplicate(self.terms[k],self.terms[p]) for p in pos)):
                    weights[i,j]=0
                elif k in candidates:
                    weights[i,j]=.25
                elif r["subject"]==e["subject"]:
                    # Unknown same-subject documents are not reliable negatives.
                    weights[i,j]=0
                elif self.query_terms[e["id"]] & self.terms[k]:
                    weights[i,j]=.25
            if i in hard and weights[i,index[hard[i]]]>0:
                scores=self.teacher(e)
                p=chosen[i]
                h=hard[i]
                # Do not distil a lexical ranking that contradicts a positive.
                if p in scores and h in scores and scores[p]>=scores[h]:
                    cols=[index[p],index[h]]
                    teacher_scores[i,cols]=torch.tensor([scores[p],scores[h]])
                    teacher_pairs.append((i,cols))
        return ids,positive.to(device),weights.to(device),teacher_scores.to(device),teacher_pairs

def run(args):
    if (args.epochs<1 or args.batch_size<2 or args.max_steps<0 or args.threads<1
            or args.lr<=0 or args.temperature<=0 or args.teacher_weight<0):
        raise ValueError("invalid training configuration")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng=random.Random(args.seed)
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    tok_path=Path(args.tokenizer)
    tokenizer=BPETokenizer.load(tok_path)
    data,manifest=read_dataset(args.dataset,tok_path,tokenizer)
    cfg=EncoderConfig(vocab_size=tokenizer.vocab_size,max_length=manifest["max_length"])
    model=UniversityEncoder(cfg).to(device)
    corpus=TrainingCorpus(*data["train"],tokenizer)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    steps_per_epoch=math.ceil(len(corpus.examples)/args.batch_size)
    if args.max_steps:
        steps_per_epoch=min(steps_per_epoch,args.max_steps)
    total=max(1,steps_per_epoch*args.epochs)
    warmup=max(1,int(total*.1))
    def schedule(step):
        if step<warmup:
            return (step+1)/warmup
        return max(.1,.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/"encoder-best.pt"
    start=perf_counter()
    initial=evaluate(model,tokenizer,*data["validation"],device=device,alphas=(0,1))
    print("initial_validation="+json.dumps(initial),flush=True)
    best=-1.0
    best_epoch=None
    history=[]
    step=0
    skipped=0
    teacher_pairs_seen=0
    trained_examples=set()
    for epoch in range(1,args.epochs+1):
        model.train()
        losses=[]
        contrast=[]
        distilled=[]
        for examples in corpus.batches(rng,args.batch_size,args.max_steps):
            ids,pos,weights,ts,pairs=corpus.prepare(examples,rng,device)
            # Avoid a zero-information batch; at least one true negative
            # must be present for an update.
            if not ((weights>0)&~pos).any():
                skipped+=1
                continue
            q=collate_texts(tokenizer,[e["query"] for e in examples],cfg,device=device)
            d=collate_texts(tokenizer,[corpus.rows[k]["encoder_text"] for k in ids],cfg,device=device)
            optimizer.zero_grad(set_to_none=True)
            logits=(model(**q)@model(**d).T)/args.temperature
            c=contrastive_loss(logits,pos,weights)
            t=teacher_loss(logits,ts,pairs)
            loss=c+args.teacher_weight*t
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step+=1
            trained_examples.update(e["id"] for e in examples)
            teacher_pairs_seen+=len(pairs)
            losses.append(loss.item())
            contrast.append(c.item())
            distilled.append(t.item())
            if step%20==0:
                print(f"step={step} loss={loss.item():.5f}",flush=True)
        if not losses:
            raise RuntimeError("no trainable batches")
        val=evaluate(model,tokenizer,*data["validation"],device=device)
        record={"epoch":epoch,"steps":step,"loss":sum(losses)/len(losses),
                "contrastive_loss":sum(contrast)/len(contrast),
                "lexical_distillation_loss":sum(distilled)/len(distilled),
                "validation":val}
        history.append(record)
        print("epoch="+json.dumps(record),flush=True)
        # Select checkpoint by dense MRR, never by held-out test.
        if val["0.0"]["mrr_at5"]>best:
            best=val["0.0"]["mrr_at5"]
            best_epoch=epoch
            save_checkpoint(checkpoint,model,tok_path,trained_steps=step)
    model,best_steps=load_checkpoint(checkpoint,tok_path,map_location=device)
    validation=evaluate(model,tokenizer,*data["validation"],device=device)
    alpha=choose_alpha(validation)
    test=evaluate(model,tokenizer,*data["test"],device=device,alphas=tuple(sorted({0.,1.,alpha})))
    report={"weights":"trained_from_scratch","teacher":"BM25F_weak_lexical",
            "dataset_sha256":hashlib.sha256((Path(args.dataset)/"dataset-manifest.json").read_bytes()).hexdigest(),
            "tokenizer_sha256":tokenizer_sha256(tok_path),
            "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "parameters":model.parameter_count(),"checkpoint_bytes":checkpoint.stat().st_size,
            "config":cfg.to_dict(),"seed":args.seed,"epochs":args.epochs,
            "batch_size":args.batch_size,"max_steps_per_epoch":args.max_steps,
            "learning_rate":args.lr,"temperature":args.temperature,"teacher_weight":args.teacher_weight,
            "total_steps":step,"best_steps":best_steps,"best_epoch":best_epoch,
            "distinct_train_queries_seen":len(trained_examples),"train_queries_available":len(corpus.examples),
            "teacher_pairs_seen":teacher_pairs_seen,"skipped_batches":skipped,
            "initial_validation":initial,"history":history,"validation":validation,
            "selected_lexical_weight":alpha,"test":test,
            "seconds":round(perf_counter()-start,3),"device":str(device),"threads":args.threads,
            "limitations":["weak labels and potential false negatives",
                           "tokenizer was fitted on full corpus",
                           "subject holdout and synthetic queries are not independent human evaluation",
                           "lexical teacher is not a pretrained semantic teacher"]}
    (out/"training-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FINAL_REPORT="+json.dumps(report,ensure_ascii=False),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",default="ai/generated/dataset")
    p.add_argument("--tokenizer",default="ai/generated/tokenizer.json")
    p.add_argument("--output",default="ai/generated/trained")
    p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--max-steps",type=int,default=0,help="cap updates attempted per epoch; 0=all batches")
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--temperature",type=float,default=.07)
    p.add_argument("--teacher-weight",type=float,default=.1)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--device",default="cpu")
    run(p.parse_args())

if __name__=="__main__":
    main()
