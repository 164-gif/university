"""Fine-tune a University encoder on the v0.3 grouped query dataset."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from ai.model.checkpoint import load_checkpoint, save_checkpoint
from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder
from ai.model.inputs import collate_texts
from ai.tokenizer.bpe import BPETokenizer


def load_rows(root, name):
    return [json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines() if line]


def run(args):
    root=Path(args.data); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    tokpath=Path(args.tokenizer); tok=BPETokenizer.load(tokpath)
    train=load_rows(root,"train-examples.jsonl"); val=load_rows(root,"validation-examples.jsonl")
    rows=load_rows(root,"passages.jsonl"); passages={r["id"]:r for r in rows}
    groups={}
    for e in train:
        if e["label"]=="review" or not e["positive_ids"]: continue
        key=e.get("parent_id") or e["id"]
        groups.setdefault(key,[]).append(e)
    if args.init:
        model,_=load_checkpoint(args.init,tokpath,map_location="cpu")
        cfg=model.config
        if cfg.vocab_size!=tok.vocab_size: raise ValueError("v0.2 tokenizer/model mismatch")
    else:
        cfg=EncoderConfig(vocab_size=tok.vocab_size,hidden_size=args.hidden,num_layers=6,
                          num_heads=8,intermediate_size=1024,embedding_size=256,max_length=512)
        model=UniversityEncoder(cfg)
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,fused=device.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=args.precision=="fp16")
    epochs=args.epochs; batch=args.batch
    steps=math.ceil(len(groups)/batch); total=steps*epochs; warm=max(1,int(total*.05))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/warm,max(.05,.5*(1+math.cos(math.pi*max(0,s-warm)/(total-warm))))))
    rng=random.Random(args.seed); best=-1; step=0
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    report={"config":config,"params":model.parameter_count(),"history":[]}
    vqueries=[x["query"] for x in val]
    for epoch in range(1,epochs+1):
        start=time.perf_counter(); model.train(); keys=list(groups); rng.shuffle(keys); losses=[]
        for offset in range(0,len(keys),batch):
            selected=[rng.choice(groups[k]) for k in keys[offset:offset+batch]]
            qtexts=[e["query"] for e in selected]
            pids=list(dict.fromkeys(pid for e in selected for pid in e["positive_ids"]))
            dtexts=[passages[pid]["encoder_text"] for pid in pids]
            q=collate_texts(tok,qtexts,cfg,device=device); d=collate_texts(tok,dtexts,cfg,device=device)
            positives=torch.zeros((len(selected),len(pids)),dtype=torch.bool,device=device)
            for i,e in enumerate(selected):
                for pid in e["positive_ids"]: positives[i,pids.index(pid)]=True
            with torch.autocast("cuda",dtype=torch.bfloat16 if args.precision=="bf16" else torch.float16,enabled=device.type=="cuda" and args.precision!="fp32"):
                qv=model(**q); dv=model(**d)
            logits=qv.float()@dv.float().T/.07
            loss=torch.nn.functional.cross_entropy(logits.masked_fill(~positives,-1e4),positives.float().argmax(1))
            if args.consistency:
                typo=next((x for x in groups.get(selected[0].get("parent_id") or selected[0]["id"],[]) if x.get("origin")=="typo"),None)
                if typo:
                    noisy=collate_texts(tok,[typo["query"]],cfg,device=device)
                    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda" and args.precision=="bf16"):
                        nv=model(**noisy)
                    loss=loss+args.consistency*(1-(qv[:1].float()*nv.float()).sum(-1)).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); scheduler.step()
            step+=1; losses.append(loss.item())
        metrics=evaluate(model,tok,val,passages,device,batch)
        rec={"epoch":epoch,"loss":sum(losses)/len(losses),"validation":metrics,"seconds":time.perf_counter()-start,"vram_peak_mib":torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else None}
        report["history"].append(rec); score=metrics["mrr_at5"]
        if score>best:
            best=score; save_checkpoint(out/"encoder-best.pt",model,tokpath,trained_steps=step,epoch=epoch,best_metrics=metrics,dataset_sha256=hashlib.sha256((root/"train-examples.jsonl").read_bytes()).hexdigest(),optimizer=opt,scheduler=scheduler,scaler=scaler,config=config,report=report)
        report.update(epoch=epoch,steps=step,best_epoch=max((x["epoch"] for x in report["history"] if x["validation"]["mrr_at5"]==best),default=epoch))
        tmp=out/"training-report.json.tmp"; tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)); tmp.replace(out/"training-report.json")
        save_checkpoint(out/"encoder-last.pt",model,tokpath,trained_steps=step,epoch=epoch,best_metrics={"mrr_at5":best},dataset_sha256=hashlib.sha256((root/"train-examples.jsonl").read_bytes()).hexdigest(),optimizer=opt,scheduler=scheduler,scaler=scaler,config=config,report=report)
        print(json.dumps(rec),flush=True)
    return report


def evaluate(model,tok,examples,passages,device,batch_size):
    model.eval(); scope={}
    rows=list(passages.values())
    doc=[]
    with torch.inference_mode():
        for i in range(0,len(rows),batch_size):
            b=collate_texts(tok,[r["encoder_text"] for r in rows[i:i+batch_size]],model.config,device=device); doc.extend(model(**b).float().cpu())
        qv=[]
        for i in range(0,len(examples),batch_size):
            b=collate_texts(tok,[r["query"] for r in examples[i:i+batch_size]],model.config,device=device); qv.extend(model(**b).float().cpu())
    idx={r["id"]:i for i,r in enumerate(rows)}; ranks=[]
    for e,q in zip(examples,qv):
        pool=[i for i,r in enumerate(rows) if r["cohort"]==e["cohort"] and r["subject"]==e["subject"]]
        order=sorted(pool,key=lambda i:-float(doc[i]@q)); pos=set(e["positive_ids"])
        ranks.append(next((k for k,i in enumerate(order,1) if rows[i]["id"] in pos),None))
    n=len(ranks)
    return {"queries":n,"top1":sum(x==1 for x in ranks)/n,"top3":sum(x and x<=3 for x in ranks)/n,"top5":sum(x and x<=5 for x in ranks)/n,"mrr_at5":sum(1/x for x in ranks if x and x<=5)/n}


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,default=Path("ai/data/v03")); p.add_argument("--output",type=Path,required=True); p.add_argument("--tokenizer",type=Path,default=Path("ai/weights/v0.2/tokenizer.json")); p.add_argument("--init"); p.add_argument("--epochs",type=int,default=50); p.add_argument("--batch",type=int,default=48); p.add_argument("--lr",type=float,default=3e-5); p.add_argument("--precision",choices=["bf16","fp16","fp32"],default="bf16"); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42); p.add_argument("--consistency",type=float,default=.1); p.add_argument("--hidden",type=int,default=256); run(p.parse_args())
