"""50-epoch training with train-only MLM, E5 distillation and variant rotation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import contextlib
from functools import lru_cache
import subprocess
from time import perf_counter

import torch

from ai.indexing.chunker import digest
from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder
from ai.model.inputs import collate_texts
from ai.model.checkpoint import save_checkpoint,load_checkpoint,tokenizer_sha256
from ai.tokenizer.bpe import BPETokenizer
from ai.training.train import read_dataset,TrainingCorpus
from ai.training.losses import contrastive_loss,teacher_loss
from ai.training.mlm import pretrain
from ai.training.semantic_teacher import SemanticTeacher,semantic_loss,MODEL,REVISION
from ai.training.retrieval_eval import evaluate,choose_alpha
from ai.training.manual_eval import evaluate_manual

def load_variants(directory,examples,tokenizer,max_length):
    p=Path(directory)
    m=json.loads((p/"augmentation-manifest.json").read_text())
    raw=(p/"train-variants.jsonl").read_bytes().decode("utf-8")
    if digest(raw)!=m["variant_sha256"] or hashlib.sha256((p/"dataset-manifest.json").read_bytes()).hexdigest()!=m["dataset_manifest_sha256"]:
        raise ValueError("variant integrity mismatch")
    variants={x["id"]:x["queries"] for x in map(json.loads,raw.splitlines())}
    if set(variants)!={e["id"] for e in examples}:
        raise ValueError("variant IDs mismatch")
    if any(not qs or any(len(tokenizer.encode(q))>max_length for q in qs) for qs in variants.values()):
        raise ValueError("invalid variants")
    return variants,m

def write_report(out,report):
    p=out/"training-report.json"
    tmp=p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(p)

def gpu_temperature(device):
    if device.type!="cuda": return None
    try:
        result=subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],
                              check=True,capture_output=True,text=True,timeout=3)
        return int(result.stdout.splitlines()[0])
    except Exception: return None

def run(args):
    if args.epochs<1 or args.mlm_epochs<0 or args.batch_size<2 or args.threads<1 or args.lr<=0 or args.grad_accum<1:
        raise ValueError("invalid training config")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng=random.Random(args.seed)
    device=torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    amp_dtype = (torch.bfloat16 if args.precision == "bf16" else
                 torch.float16 if args.precision == "fp16" else None)
    if amp_dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but this CUDA device does not support it")
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.precision == "fp16"))
    root=Path(args.directory)
    tok_path=root/"tokenizer.json"
    fit=json.loads((root/"tokenizer-report.json").read_text())
    if fit["fit_scope"]!="train_subjects_only" or fit["tokenizer_sha256"]!=tokenizer_sha256(tok_path):
        raise ValueError("expected train-only BPE")
    tokenizer=BPETokenizer.load(tok_path)
    base_encode=tokenizer.encode
    @lru_cache(maxsize=120000)
    def encoded_once(text): return tuple(base_encode(text))
    tokenizer.encode=lambda text:list(encoded_once(text))
    data,manifest=read_dataset(root/"dataset",tok_path,tokenizer)
    cfg=EncoderConfig(vocab_size=tokenizer.vocab_size,max_length=manifest["max_length"])
    train_rows,train_examples=data["train"]
    variants,aug=load_variants(root/"dataset",train_examples,tokenizer,cfg.max_length)
    if set(r["subject"] for r in train_rows)&set(r["subject"] for s in ("validation","test") for r in data[s][0]):
        raise ValueError("subject leakage")
    model=UniversityEncoder(cfg).to(device)
    out=root/"trained"
    out.mkdir(parents=True,exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard=SummaryWriter(log_dir=out/"tensorboard")
    except ImportError:
        tensorboard=None
    epoch_log=out/"training.jsonl"
    start=perf_counter()
    report={"status":"running","epochs_requested":args.epochs,"mlm_epochs_requested":args.mlm_epochs,
            "seed":args.seed,"teacher":MODEL if not args.no_teacher else None,
            "teacher_revision":REVISION if not args.no_teacher else None,
            "training_parameters":{"device":str(device),"precision":args.precision,
                "batch_size":args.batch_size,"gradient_accumulation":args.grad_accum,
                "learning_rate":args.lr,"semantic_weight":args.semantic_weight,
                "optimizer":"AdamW fused when supported","gradient_clip_norm":1.0,
                "mlm_mask_probability":0.15,"torch_compile_requested":args.compile},
            "config":cfg.to_dict(),"parameters":model.parameter_count(),
            "tokenizer_sha256":tokenizer_sha256(tok_path),"dataset_sha256":aug["dataset_manifest_sha256"],
            "variant_count":aug["variant_count"],"base_queries":len(train_examples),
            "augmentation":aug["augmentation"],"tokenization_cache_capacity":120000,"history":[]}
    write_report(out,report)
    report["initial_validation"]=evaluate(model,tokenizer,*data["validation"],device=device,alphas=(0,1))
    report["mlm"]=pretrain(model,tokenizer,train_rows,epochs=args.mlm_epochs,batch_size=args.batch_size,
                           device=device,seed=args.seed,lr=args.lr)
    if report["mlm"]:
        mlm_steps=sum(row["steps"] for row in report["mlm"])
        save_checkpoint(out/"encoder-mlm.pt",model,tok_path,trained_steps=mlm_steps,
                        epoch=args.mlm_epochs,dataset_sha256=aug["dataset_manifest_sha256"],
                        config={"phase":"mlm","epochs":args.mlm_epochs,"masking_probability":.15})
    report["post_mlm_validation"]=evaluate(model,tokenizer,*data["validation"],device=device,alphas=(0,1))
    write_report(out,report)
    teacher=None if args.no_teacher else SemanticTeacher(train_rows,train_examples,device,cache_dir=root/"teacher-cache")
    corpus=TrainingCorpus(train_rows,train_examples,tokenizer)
    try:
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,
                                    fused=(device.type == "cuda"))
    except (TypeError, RuntimeError):
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    total=math.ceil(math.ceil(len(train_examples)/args.batch_size)/args.grad_accum)*args.epochs
    warmup=max(1,int(total*.05))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:
        (step+1)/warmup if step<warmup else max(.05,.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)))))
    # Shuffle each variant list once; every base query is visited once per
    # epoch. Rotate rather than multiply the epoch size by ~17.
    for qs in variants.values():
        rng.shuffle(qs)
    step=0
    best=-1.
    best_epoch=0
    seen=set()
    checkpoint=out/"encoder-best.pt"
    last_checkpoint=out/"encoder-last.pt"
    first_epoch=1
    if args.resume:
        state=torch.load(last_checkpoint,map_location="cpu",weights_only=False)
        if state.get("tokenizer_sha256")!=tokenizer_sha256(tok_path) or state.get("dataset_sha256")!=aug["dataset_manifest_sha256"]:
            raise ValueError("resume checkpoint tokenizer/dataset hash mismatch")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"): scaler.load_state_dict(state["scaler"])
        step=state["trained_steps"]; first_epoch=state["epoch"]+1
        best=state.get("best_metrics",{}).get("mrr_at5",-1.); best_epoch=state.get("best_metrics",{}).get("epoch",0)
        torch.set_rng_state(state["rng"]["torch"])
        if device.type=="cuda" and state["rng"].get("cuda"): torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        if state["rng"].get("python"): rng.setstate(state["rng"]["python"])
        report.update(state.get("report",{}))
    if args.compile:
        try:
            candidate=torch.compile(model)
            # Compilation may fail lazily at first invocation; retain the
            # eager model as the authoritative parameter/checkpoint owner.
            model=candidate
        except Exception as exc:
            print(f"torch.compile unavailable: {type(exc).__name__}: {exc}",flush=True)
    for epoch in range(first_epoch,args.epochs+1):
        epoch_start=perf_counter()
        model.train()
        losses=[]
        contrasts=[]
        lexical=[]
        semantic=[]
        for examples in corpus.batches(rng,args.batch_size,0):
            ids,pos,w,ts,pairs=corpus.prepare(examples,rng,device)
            query_texts=[variants[e["id"]][(epoch-1)%len(variants[e["id"]])] for e in examples]
            q=collate_texts(tokenizer,query_texts,cfg,device=device)
            d=collate_texts(tokenizer,[corpus.rows[k]["encoder_text"] for k in ids],cfg,device=device)
            if (len(losses) % args.grad_accum)==0: optimizer.zero_grad(set_to_none=True)
            autocast=(torch.autocast(device_type="cuda",dtype=amp_dtype) if amp_dtype and device.type=="cuda" else contextlib.nullcontext())
            with autocast:
                logits=(model(**q)@model(**d).T)/.07
            if teacher:
                scores=teacher.scores(examples,ids,device)
                best_positive=scores.masked_fill(~pos,-torch.inf).max(1,keepdim=True).values
                # Potential false negatives close to a known positive in
                # teacher space are downweighted, not treated as truth.
                w=torch.where((~pos)&(scores>=best_positive-.015),w*.1,w)
                sem=semantic_loss(logits,scores,pos,w)
            else:
                sem=logits.sum()*0
            c_loss=contrastive_loss(logits,pos,w)
            l_loss=.1*teacher_loss(logits,ts,pairs)
            loss=c_loss+l_loss+args.semantic_weight*sem
            loss=loss/args.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            scaler.scale(loss).backward()
            if (len(losses)+1)%args.grad_accum==0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); scheduler.step()
            step+=1
            seen.update((e["id"],qtext) for e,qtext in zip(examples,query_texts))
            losses.append(loss.item()*args.grad_accum)
            contrasts.append(c_loss.item()); lexical.append(l_loss.item())
            semantic.append(sem.item())
        if len(losses)%args.grad_accum:
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); scheduler.step()
        val=evaluate(model,tokenizer,*data["validation"],device=device)
        epoch_seconds=perf_counter()-epoch_start
        vram=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else None
        temperature=gpu_temperature(device)
        record={"epoch":epoch,"steps":step,"loss":sum(losses)/len(losses),
                "contrastive_loss":sum(contrasts)/len(contrasts),
                "lexical_distillation_loss":sum(lexical)/len(lexical),
                "semantic_loss":sum(semantic)/len(semantic),
                "learning_rate":optimizer.param_groups[0]["lr"],
                "queries_per_second":len(train_examples)/max(epoch_seconds,1e-9),
                "epoch_seconds":epoch_seconds,"vram_peak_mib":vram,"gpu_temperature_c":temperature,
                "validation":val}
        report["history"].append(record)
        score=val["0.0"]["mrr_at5"]
        if score>best:
            best=score
            best_epoch=epoch
            save_checkpoint(checkpoint,model._orig_mod if hasattr(model,"_orig_mod") else model,
                            tok_path,trained_steps=step,epoch=epoch,
                            best_metrics={"mrr_at5":best,"epoch":best_epoch},
                            dataset_sha256=aug["dataset_manifest_sha256"],
                            config=vars(args),report=report,python_rng=rng)
        report.update({"epochs_completed":epoch,"best_epoch":best_epoch,"total_steps":step,
                       "distinct_query_variants_seen":len(seen),"tokenized_sequences_cached":encoded_once.cache_info().currsize,
                       "elapsed_seconds":round(perf_counter()-start,2),
                       "estimated_seconds_remaining":(perf_counter()-start)/epoch*max(0,args.epochs-epoch),
                       "best_checkpoint":str(checkpoint),"best_validation_mrr_at5":best})
        write_report(out,report)
        save_checkpoint(last_checkpoint,model._orig_mod if hasattr(model,"_orig_mod") else model,tok_path,trained_steps=step,optimizer=optimizer,scheduler=scheduler,
                        scaler=scaler,epoch=epoch,best_metrics={"mrr_at5":best,"epoch":best_epoch},
                        dataset_sha256=aug["dataset_manifest_sha256"],python_rng=rng,
                        config=vars(args),report=report)
        with epoch_log.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+"\n")
        if tensorboard:
            for key in ("loss","contrastive_loss","lexical_distillation_loss","semantic_loss","learning_rate",
                        "queries_per_second","epoch_seconds","vram_peak_mib","gpu_temperature_c"):
                if record.get(key) is not None: tensorboard.add_scalar("train/"+key,record[key],epoch)
            for key in ("top1","top5","mrr_at5"):
                tensorboard.add_scalar("validation/"+key,val["0.0"][key],epoch)
            tensorboard.flush()
        print("EPOCH="+json.dumps(record),flush=True)
    model,best_steps=load_checkpoint(checkpoint,tok_path,map_location=device)
    val=evaluate(model,tokenizer,*data["validation"],device=device)
    alpha=choose_alpha(val)
    report["validation"]=val
    report["selected_lexical_weight"]=alpha
    report["test"]=evaluate(model,tokenizer,*data["test"],device=device,alphas=tuple(sorted({0.,1.,alpha})))
    cases=json.loads((Path(__file__).parents[1]/"evaluation"/"paraphrases_test.json").read_text(encoding="utf-8"))
    report["manual_test"]=evaluate_manual(model,tokenizer,data["test"][0],cases,device=device,alpha=alpha)
    report.update({"status":"completed","best_steps":best_steps,"checkpoint_bytes":checkpoint.stat().st_size,
                   "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "seconds":round(perf_counter()-start,2)})
    write_report(out,report)
    if tensorboard: tensorboard.close()
    print("FINAL="+json.dumps({k:v for k,v in report.items() if k!="history"},ensure_ascii=False),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--directory",default="ai/generated/full")
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--mlm-epochs",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--semantic-weight",type=float,default=.5)
    p.add_argument("--device",default="cuda")
    p.add_argument("--precision",choices=("bf16","fp16","fp32"),default="bf16")
    p.add_argument("--grad-accum",type=int,default=1)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--compile",action="store_true")
    p.add_argument("--no-teacher",action="store_true",help="local debugging only")
    run(p.parse_args())
