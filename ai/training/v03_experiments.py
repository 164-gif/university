"""Train and resume University v0.3 retrieval experiments on CUDA."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import shutil
from time import perf_counter

import torch
from torch.nn import functional as F

from ai.model.checkpoint import load_checkpoint, save_checkpoint, tokenizer_sha256
from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder
from ai.model.inputs import collate_texts
from ai.search.bm25 import BM25
from ai.tokenizer.bpe import BPETokenizer, EOS, train
from ai.training.losses import contrastive_loss
from ai.training.mlm import pretrain
from ai.training.semantic_teacher import SemanticTeacher, MODEL, REVISION
from ai.training.v03_data import category, prepare, scope, write_json
from ai.training.v03_eval import Ranking, encode_rank, fuse

BASE = "intfloat/multilingual-e5-base"
BASE_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
V02 = Path("ai/weights/v0.2")


def device_for_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for v0.3 training")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def cached_tokenizer(path):
    tokenizer = BPETokenizer.load(path)
    encode = tokenizer.encode

    @lru_cache(maxsize=40000)
    def bounded(text):
        ids = encode(text)
        return ids if len(ids)<=512 else ids[:511]+[EOS]

    tokenizer.encode = bounded
    return tokenizer


def data_and_teacher(args, device):
    bank = prepare(args.data, args.legacy, args.work)
    train_docs = bank.train_docs
    train_examples = bank.train_examples
    teacher_model, revision = (BASE, BASE_REVISION) if args.teacher=="base" else (MODEL, REVISION)
    teacher = SemanticTeacher(train_docs, train_examples, device, args.work/"teacher"/args.teacher,
                              model_name=teacher_model,revision=revision)
    val_rows = [r for r in bank.rows.values() if r["split"]=="validation"]
    val_examples = bank.examples["validation"]
    ranking = Ranking(val_rows, val_examples)
    teacher_val = SemanticTeacher(val_rows,val_examples,device,args.work/"teacher-validation"/args.teacher,
                                  model_name=teacher_model,revision=revision)
    teacher_metrics = ranking.summaries(ranking.dense(teacher_val.queries,teacher_val.docs)[0],detailed=True)
    path = args.work/f"teacher-{args.teacher}.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    write_json(path,{"model":teacher_model,"revision":revision,"license":"MIT",
               "train_peak_allocated_mib":getattr(teacher,"peak_mib",previous.get("train_peak_allocated_mib")),
               "validation_peak_allocated_mib":getattr(teacher_val,"peak_mib",previous.get("validation_peak_allocated_mib")),
               "validation":teacher_metrics})
    return bank, teacher, ranking


def mine(bank, teacher):
    docs = bank.train_docs
    examples = bank.train_examples
    bm25 = BM25(docs)
    candidates = {}
    pools = defaultdict(list)
    for i, doc in enumerate(docs):
        pools[scope(doc)].append(i)
    scores = (teacher.queries @ teacher.docs.T).tolist()
    for row, similarity in zip(examples, scores):
        positive = set(row["positive_ids"])
        pos_score = max(similarity[teacher.doc_ids[key]] for key in positive)
        # Similar teacher scores and an identical originating section are ambiguous.
        sections = {(bank.rows[key]["source"],tuple(bank.rows[key]["headings"])) for key in positive}
        ranked = sorted(pools[scope(row)], key=lambda i:-similarity[i])
        lexical = bm25.search(row["query"],subject=row["subject"],cohort=row["cohort"],limit=16,fuzzy=False)
        mixed = dict.fromkeys([hit.passage["id"] for hit in lexical]+[docs[i]["id"] for i in ranked[:30]])
        candidates[row["id"]] = [key for key in mixed if key not in positive
            and similarity[teacher.doc_ids[key]] < pos_score-.018
            and (bank.rows[key]["source"],tuple(bank.rows[key]["headings"])) not in sections][:12]
    return candidates


def choose_queries(rows, families, ratio, rng):
    eligible = [row["id"] for row in rows if families.get(row["id"])]
    chosen = set(rng.sample(eligible,min(len(eligible),round(ratio*len(rows)))))
    return [rng.choice(families[row["id"]])["query"] if row["id"] in chosen else row["query"]
            for row in rows]


def smart_batches(rows, tokenizer, docs, batch_size, rng):
    order = list(rows)
    rng.shuffle(order)
    batches = []
    for offset in range(0,len(order),batch_size*12):
        block = sorted(order[offset:offset+batch_size*12], key=lambda row:
                       max(len(tokenizer.encode(docs[key]["encoder_text"])) for key in row["positive_ids"]))
        batches.extend(block[i:i+batch_size] for i in range(0,len(block),batch_size))
    rng.shuffle(batches)
    return batches


def retrieval_loss(model,tok,bank,teacher,neighbors,rows,inputs,args,device,rng):
    cfg = model.config
    ids = list(dict.fromkeys(key for row in rows for key in row["positive_ids"]))
    for row in rows:
        candidates = neighbors[row["id"]]
        if candidates:
            ids.append(rng.choice(candidates))
    ids = list(dict.fromkeys(ids))
    positions = {key:i for i,key in enumerate(ids)}
    positive = torch.zeros((len(rows),len(ids)),dtype=torch.bool)
    for i,row in enumerate(rows):
        positive[i,[positions[key] for key in row["positive_ids"]]]=True
    positive = positive.to(device)
    doc = model(**collate_texts(tok,[bank.rows[k]["encoder_text"] for k in ids],cfg,device=device))
    query = model(**collate_texts(tok,inputs,cfg,device=device))
    logits = (query @ doc.T).float()/.07
    semantic = teacher.scores(rows,ids,device)
    best = semantic.masked_fill(~positive,-torch.inf).max(dim=1,keepdim=True).values
    confidence = torch.where(semantic>best-.018,.10,1.)
    confidence = torch.where(positive,1.,confidence)
    retrieval = contrastive_loss(logits,positive,confidence,per_query=True)
    target = F.softmax(semantic/.035,dim=1)
    distillation = F.kl_div(F.log_softmax(logits,dim=1),target,reduction="none").sum(dim=1)
    confidence = torch.tensor([args.weak_weight if category(row)=="automatic" else 1. for row in rows],device=device)
    loss = ((retrieval + args.alpha*distillation)*confidence).sum()/confidence.sum()
    pairs = [(i,row["query"]) for i,row in enumerate(rows) if row["query"]!=inputs[i]]
    if pairs and args.beta:
        clean = model(**collate_texts(tok,[text for _,text in pairs],cfg,device=device))
        noisy = query[[i for i,_ in pairs]]
        loss = loss + args.beta*(1-(clean*noisy).sum(-1)).mean()
    return loss


def run(args):
    device = device_for_cuda()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    bank, teacher, validation = data_and_teacher(args, device)
    if args.kind=="teacher":
        return
    neighbors = mine(bank,teacher)
    lexical,_ = validation.lexical(True)
    tokenizer_path = out/"tokenizer.json"
    if args.kind=="a":
        shutil.copyfile(V02/"tokenizer.json",tokenizer_path)
        model,_ = load_checkpoint(V02/"encoder-best.pt",tokenizer_path,map_location=device)
    else:
        fitted_path = args.work/"train-only-tokenizer.json"
        if not fitted_path.exists():
            source = [r["encoder_text"] for r in bank.train_docs]+[r["query"] for r in bank.train_examples]
            train(source,vocab_size=16384).save(fitted_path)
        shutil.copyfile(fitted_path,tokenizer_path)
        tokenizer = BPETokenizer.load(tokenizer_path)
        config = EncoderConfig(vocab_size=tokenizer.vocab_size,hidden_size=256,num_layers=6,num_heads=8,
                               intermediate_size=1024,embedding_size=256)
        model = UniversityEncoder(config).to(device)
    tokenizer = cached_tokenizer(tokenizer_path)
    settings = {key:(str(value) if isinstance(value,Path) else value)
                for key,value in vars(args).items() if key!="resume"}
    settings["dataset_sha256"] = bank.fingerprint
    report = {"config":settings,"parameters":model.parameter_count(),"dataset":bank.audit,"history":[]}
    last = out/"encoder-last.pt"
    if args.kind=="b" and not args.resume:
        mlm_path = out/"encoder-mlm.pt"
        if mlm_path.exists():
            model,_ = load_checkpoint(mlm_path,tokenizer_path,map_location=device)
        else:
            report["mlm"] = pretrain(model,tokenizer,bank.train_docs,epochs=5,batch_size=args.batch,
                                     device=device,seed=args.seed,lr=args.lr,amp=True)
            save_checkpoint(mlm_path,model,tokenizer_path,
                            trained_steps=sum(row["steps"] for row in report["mlm"]),
                            dataset_sha256=bank.fingerprint,report=report)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,fused=True)
    per_epoch = math.ceil(len(bank.train_examples)/args.batch)
    total = args.epochs*math.ceil(per_epoch/args.accum)
    warmup = max(1,total//20)
    def schedule(step):
        if step<warmup:
            return (step+1)/warmup
        return max(.05,.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,schedule)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)
    step, start_epoch, best = 0, 1, -1.
    if args.resume:
        state = torch.load(last,map_location="cpu",weights_only=True)
        if state["training_config"]!=settings or state["tokenizer_sha256"]!=tokenizer_sha256(tokenizer_path):
            raise ValueError("resume configuration or data differs")
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        rng.setstate(state["rng"]["python"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        step,start_epoch = state["trained_steps"],state["epoch"]+1
        report,best = state["report"],state["best_metrics"]["selection_score"]
    for epoch in range(start_epoch,args.epochs+1):
        model.train()
        start = perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        batches = smart_batches(bank.train_examples,tokenizer,bank.rows,args.batch,rng)
        selected = choose_queries(bank.train_examples,bank.families,args.typo_ratio,rng)
        samples = {row["id"]:query for row,query in zip(bank.train_examples,selected)}
        total_loss = 0.
        opt.zero_grad(set_to_none=True)
        for i, rows in enumerate(batches):
            with torch.autocast("cuda",dtype=dtype):
                loss = retrieval_loss(model,tokenizer,bank,teacher,neighbors,rows,
                                      [samples[row["id"]] for row in rows],args,device,rng)
            group_size = min(args.accum,len(batches)-(i//args.accum)*args.accum)
            scaler.scale(loss/group_size).backward()
            if (i+1)%args.accum==0 or i+1==len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=dtype==torch.bfloat16)
                previous_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale()>=previous_scale:
                    scheduler.step()
                    step += 1
                opt.zero_grad(set_to_none=True)
            total_loss += loss.detach().item()
        order, scores = encode_rank(model,tokenizer,validation,device,64)
        metrics = validation.summaries(order,detailed=True)
        # Evaluate retrieval independently from lexical-only configurations.
        hybrid = validation.summaries(fuse(validation,lexical,order,scores,"weighted",.75))
        selection = metrics["group_mean"]["mrr_at5"]
        record = {"epoch":epoch,"loss":total_loss/len(batches),"validation":metrics,
                  "validation_hybrid":hybrid,"seconds":perf_counter()-start,
                  "peak_mib":torch.cuda.max_memory_allocated(device)/2**20,
                  "lr":opt.param_groups[0]["lr"],"steps":step}
        report["history"].append(record)
        if selection>best:
            best = selection
            report["best_epoch"] = epoch
            save_checkpoint(out/"encoder-best.pt",model,tokenizer_path,trained_steps=step,epoch=epoch,
                            best_metrics=metrics,dataset_sha256=bank.fingerprint,config=settings)
        save_checkpoint(last,model,tokenizer_path,trained_steps=step,epoch=epoch,optimizer=opt,
                        scheduler=scheduler,scaler=scaler,python_rng=rng,dataset_sha256=bank.fingerprint,
                        best_metrics={"selection_score":best},config=settings,report=report)
        write_json(out/"training-report.json",report)
        print(f"{out.name} epoch={epoch}/{args.epochs} loss={record['loss']:.4f} "
              f"val_mrr={selection:.4f} best={best:.4f} seconds={record['seconds']:.1f} "
              f"peak_mib={record['peak_mib']:.0f}",flush=True)


def parser(include_kind=True):
    p = argparse.ArgumentParser(description=__doc__)
    if include_kind:
        p.add_argument("kind",choices=("a","b","teacher"))
    p.add_argument("--data",type=Path,default=Path("ai/data/v03"))
    p.add_argument("--legacy",type=Path,default=Path("ai/generated/full/dataset/test-examples.jsonl"))
    p.add_argument("--work",type=Path,default=Path("ai/generated/v03"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--teacher",choices=("small","base"),default="small")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch",type=int,default=48)
    p.add_argument("--accum",type=int,default=1)
    p.add_argument("--lr",type=float,default=3e-5)
    p.add_argument("--typo-ratio",type=float,default=.4)
    p.add_argument("--weak-weight",type=float,default=.45)
    p.add_argument("--alpha",type=float,default=.6)
    p.add_argument("--beta",type=float,default=.15)
    p.add_argument("--resume",action="store_true")
    return p


if __name__=="__main__":
    run(parser().parse_args())
