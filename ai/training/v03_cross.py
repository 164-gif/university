"""Train the compact cross-encoder and score a fixed set of candidates."""
import hashlib
import json
import math
import random
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from ai.model.checkpoint import save_checkpoint, tokenizer_sha256
from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder
from ai.training.v03_data import category, write_json
from ai.training.v03_eval import Ranking
from ai.training.v03_experiments import (
    V02, cached_tokenizer, choose_queries, data_and_teacher, device_for_cuda, mine, parser,
)

CROSS_TEACHER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CROSS_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
CROSS_SHA256 = "5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4"


class CrossEncoder(UniversityEncoder):
    def __init__(self, config):
        super().__init__(config)
        del self.projection
        self.relevance = nn.Linear(config.hidden_size,1)
        self._initialize(self.relevance)

    def forward(self,input_ids,attention_mask):
        states = self.hidden_states(input_ids,attention_mask)
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        pooled = (states*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
        return self.relevance(pooled).squeeze(-1)


def pair_batch(pairs, tokenizer, device):
    # EOS separates the full question from a right-truncated candidate. Keep BOS/EOS.
    tokens = []
    for query, passage in pairs:
        qids = tokenizer.encode(query)
        dids = tokenizer.encode(passage)
        qids = qids if len(qids)<=256 else qids[:255]+qids[-1:]
        ids = qids + dids[1:-1][:511-len(qids)] + dids[-1:]
        tokens.append(ids)
    length = max(map(len,tokens))
    padded = torch.tensor([ids+[0]*(length-len(ids)) for ids in tokens]).pin_memory().to(device,non_blocking=True)
    return {"input_ids":padded,"attention_mask":padded.ne(0)}


def predict(model,tokenizer,pairs,device,batch):
    order = sorted(range(len(pairs)),key=lambda i:sum(len(tokenizer.encode(text)) for text in pairs[i]))
    scores = torch.empty(len(pairs))
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
        for start in range(0,len(order),batch):
            idx = order[start:start+batch]
            scores[idx] = model(**pair_batch([pairs[i] for i in idx],tokenizer,device)).float().cpu()
    return scores


def rerank(model, tokenizer, ranking, candidates, device, batch):
    pairs = [(example["query"],ranking.rows[j]["encoder_text"])
             for example,ids in zip(ranking.examples,candidates) for j in ids]
    scores = predict(model,tokenizer,pairs,device,batch).tolist()
    start = 0
    order = []
    for ids in candidates:
        indexed = sorted(enumerate(ids),key=lambda item:-scores[start+item[0]])
        order.append([j for _,j in indexed])
        start += len(ids)
    return order


def teacher_scores(pairs,path,device,local=None):
    fingerprint = hashlib.sha256(json.dumps(pairs,ensure_ascii=False).encode()).hexdigest()
    if path.exists():
        state = torch.load(path,weights_only=True)
        if (state["fingerprint"],state["revision"])==(fingerprint,CROSS_REVISION):
            return state["scores"],state["peak_mib"]
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.cuda.reset_peak_memory_stats(device)
    model_id = CROSS_TEACHER
    options = {"revision":CROSS_REVISION,"trust_remote_code":False}
    if local is not None:
        with (local/"model.safetensors").open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest()!=CROSS_SHA256:
                raise ValueError("cross teacher SHA-256 differs from pinned revision")
        model_id = str(local)
        options["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_id,**options)
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
              use_safetensors=True,torch_dtype=torch.float16,**options).to(device).eval()
    scores = torch.empty(len(pairs))
    print(f"Cross teacher scoring {len(pairs)} pairs",flush=True)
    order = sorted(range(len(pairs)),key=lambda i:sum(map(len,pairs[i])))
    with torch.inference_mode():
        for start in range(0,len(order),32):
            idx = order[start:start+32]
            batch = tokenizer([pairs[i][0] for i in idx],[pairs[i][1] for i in idx],
                              padding=True,truncation=True,max_length=512,return_tensors="pt").to(device)
            scores[idx] = model(**batch).logits.flatten().float().cpu()
    peak = torch.cuda.max_memory_allocated(device)/2**20
    if not scores.isfinite().all():
        raise RuntimeError("Cross-encoder teacher returned nonfinite scores")
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save({"fingerprint":fingerprint,"revision":CROSS_REVISION,"scores":scores,"peak_mib":peak},temp)
    temp.replace(path)
    del model
    torch.cuda.empty_cache()
    return scores,peak


def load_cross(path,tokenizer_path,device):
    checkpoint = torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint["tokenizer_sha256"]!=tokenizer_sha256(tokenizer_path):
        raise ValueError("cross-encoder tokenizer SHA mismatch")
    model = CrossEncoder(EncoderConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def run(args):
    device = device_for_cuda()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    bank, teacher, _ = data_and_teacher(args,device)
    negatives = {key:ids[:3] for key,ids in mine(bank,teacher).items()}
    rows = [row for row in bank.train_examples if negatives[row["id"]]]
    pairs = {(row["id"],key):(row["query"],bank.rows[key]["encoder_text"])
             for row in rows for key in [*row["positive_ids"],*negatives[row["id"]]]}
    values, train_peak = teacher_scores(list(pairs.values()),args.work/"cross-teacher-train.pt",device,args.local_cross_teacher)
    scores = dict(zip(pairs,values.tolist()))
    clean_val = [r for r in bank.examples["validation"] if r["origin"]!="typo"]
    val = Ranking([r for r in bank.rows.values() if r["split"]=="validation"],clean_val)
    lexical,_ = val.lexical(True)
    candidates = [list(items)[:10] for items in lexical]
    val_pairs = [(row["query"],val.rows[j]["encoder_text"])
                 for row,ids in zip(clean_val,candidates) for j in ids]
    val_teacher, val_peak = teacher_scores(val_pairs,args.work/"cross-teacher-validation.pt",device,args.local_cross_teacher)
    offset = 0
    ranked = []
    for ids in candidates:
        ranked.append([j for i,j in sorted(enumerate(ids),key=lambda item:-val_teacher[offset+item[0]])])
        offset += len(ids)
    path = out/"tokenizer.json"
    path.write_bytes((V02/"tokenizer.json").read_bytes())
    tok = cached_tokenizer(path)
    config = EncoderConfig(vocab_size=tok.vocab_size)
    model = CrossEncoder(config).to(device)
    state = torch.load(V02/"encoder-best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict({key:val for key,val in state["state_dict"].items() if not key.startswith("projection.")},
                          strict=False)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,fused=True)
    total_steps = math.ceil(len(rows)/args.batch)*args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lambda s:
                 min(1.,(s+1)/max(1,total_steps*.05))*max(.05,.5*(1+math.cos(math.pi*s/total_steps))))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!="resume"}
    settings["dataset_sha256"] = bank.fingerprint
    last = out/"cross-last.pt"
    report = {"config":settings,"parameters":model.parameter_count(),"train_groups":len(rows),"history":[],
              "teacher":{"model":CROSS_TEACHER,"revision":CROSS_REVISION,"license":"Apache-2.0",
                         "train_peak_mib":train_peak,"validation_peak_mib":val_peak,
                         "validation":val.summaries(ranked),"bm25_validation":val.summaries(candidates)}}
    start_epoch, best, step = 1, -1., 0
    if args.resume:
        ck = torch.load(last,map_location="cpu",weights_only=True)
        if ck["training_config"]!=settings or ck["tokenizer_sha256"]!=tokenizer_sha256(path):
            raise ValueError("cross resume configuration differs")
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        rng.setstate(ck["rng"]["python"])
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        start_epoch, best, step = ck["epoch"]+1,ck["best_metrics"]["selection_score"],ck["trained_steps"]
        report = ck["report"]
    for epoch in range(start_epoch,args.epochs+1):
        start = perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        texts = choose_queries(rows,bank.families,args.typo_ratio,rng)
        samples = []
        for row,query in zip(rows,texts):
            positive = rng.choice(row["positive_ids"])
            negative = rng.choice(negatives[row["id"]])
            margin = scores[row["id"],positive]-scores[row["id"],negative]
            weight = args.weak_weight if category(row)=="automatic" else 1.
            if margin<0:
                weight *= .35
            samples.append((query, positive, negative, margin,weight))
        rng.shuffle(samples)
        batches = []
        for start_batch in range(0,len(samples),args.batch*12):
            block = samples[start_batch:start_batch+args.batch*12]
            block.sort(key=lambda x:max(len(tok.encode(bank.rows[j]["encoder_text"])) for j in x[1:3]))
            batches.extend(block[pos:pos+args.batch] for pos in range(0,len(block),args.batch))
        rng.shuffle(batches)
        loss_sum = 0.
        for block in batches:
            pairs = [(query,bank.rows[key]["encoder_text"])
                     for query,positive,negative,_,_ in block for key in (positive,negative)]
            with torch.autocast("cuda",dtype=dtype):
                preds = model(**pair_batch(pairs,tok,device)).float().reshape(-1,2)
            differences = preds[:,0]-preds[:,1]
            weights = torch.tensor([x[4] for x in block],device=device)
            margins = torch.tensor([x[3] for x in block],device=device)
            hard = F.softplus(-differences)
            soft = F.binary_cross_entropy_with_logits(differences,(margins/2).sigmoid(),reduction="none")
            loss = ((hard+args.alpha*soft)*weights).sum()/weights.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale()>=old_scale:
                scheduler.step()
                step += 1
            loss_sum += loss.item()
        metrics = val.summaries(rerank(model,tok,val,candidates,device,64))
        selection = metrics["group_mean"]["mrr_at5"]
        record = {"epoch":epoch,"validation":metrics,"loss":loss_sum/len(batches),
                  "seconds":perf_counter()-start,"peak_mib":torch.cuda.max_memory_allocated(device)/2**20,"steps":step}
        report["history"].append(record)
        if selection>best:
            best = selection
            report["best_epoch"] = epoch
            save_checkpoint(out/"cross-best.pt",model,path,trained_steps=step,epoch=epoch,config=settings,
                            dataset_sha256=bank.fingerprint,best_metrics=metrics)
        save_checkpoint(last,model,path,trained_steps=step,epoch=epoch,best_metrics={"selection_score":best},
                        optimizer=opt,scheduler=scheduler,scaler=scaler,python_rng=rng,
                        dataset_sha256=bank.fingerprint,config=settings,report=report)
        write_json(out/"training-report.json",report)
        print(f"cross epoch={epoch}/{args.epochs} mrr={selection:.4f} best={best:.4f} seconds={record['seconds']:.1f}",flush=True)


if __name__=="__main__":
    options = parser(include_kind=False)
    # Cross-encoder uses the same training controls and dataset.
    options.add_argument("--local-cross-teacher",type=Path)
    run(options.parse_args())
