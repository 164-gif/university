"""Test-set comparison for BM25F, trained University Encoder and E5.

RRF is selected on Validation only. Test is touched once after that choice.
The v0.1 checkpoint is optional because this repository never committed one.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from transformers import AutoModel, AutoTokenizer

from ai.model.checkpoint import load_checkpoint
from ai.model.inputs import collate_texts
from ai.search.bm25 import BM25
from ai.tokenizer.bpe import BPETokenizer
from ai.training.semantic_teacher import MODEL, REVISION


def read_split(directory, split):
    def read(name):
        return [json.loads(line) for line in (directory/f"{split}-{name}.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return read("passages"), read("examples")


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def university_docs(model, tokenizer, rows, device, batch_size=64):
    result=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            batch=collate_texts(tokenizer,[r["encoder_text"] for r in rows[start:start+batch_size]],model.config,device=device)
            result.extend(model(**batch).float().cpu())
    return torch.stack(result)


def university_query(model, tokenizer, query, device):
    batch=collate_texts(tokenizer,[query],model.config,device=device)
    with torch.inference_mode():
        return model(**batch)[0].float().cpu()


def e5_encode(model, tokenizer, texts, device, prefix, batch_size=32):
    out=[]
    with torch.inference_mode():
        for start in range(0,len(texts),batch_size):
            batch=tokenizer([prefix+t for t in texts[start:start+batch_size]],padding=True,truncation=True,
                            max_length=512,return_tensors="pt")
            batch={k:v.to(device) for k,v in batch.items()}
            hidden=model(**batch).last_hidden_state
            mask=batch["attention_mask"].unsqueeze(-1)
            vec=torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
            out.extend(vec.float().cpu())
    return torch.stack(out)


def ranks_for(rows, examples, dense_vectors, bm25, query_vectors=None):
    scopes={}
    for i,row in enumerate(rows):
        scopes.setdefault((row["cohort"],row["subject"]),[]).append(i)
    result=[]
    for qi,example in enumerate(examples):
        indices=scopes[(example["cohort"],example["subject"])]
        dense_scores=(dense_vectors[indices]@query_vectors[qi]).tolist()
        dense=[rows[i]["id"] for i,_ in sorted(zip(indices,dense_scores),key=lambda item:(-item[1],rows[item[0]]["id"]))]
        lexical=[hit.passage["id"] for hit in bm25.search(example["query"],cohort=example["cohort"],
                  subject=example["subject"],limit=max(1,len(indices)))]
        result.append((dense,lexical,set(example["positive_ids"])))
    return result


def fused(dense, lexical, alpha, limit=10, k=20):
    if alpha==0: return dense[:limit]
    if alpha==1: return lexical[:limit]
    scores={}
    for order,key in enumerate(lexical,1): scores[key]=scores.get(key,0)+alpha/(k+order)
    for order,key in enumerate(dense,1): scores[key]=scores.get(key,0)+(1-alpha)/(k+order)
    return sorted(scores,key=lambda key:(-scores[key],key))[:limit]


def score_ranks(rankings, alpha=0):
    top1=top3=top5=mrr=recall=ndcg=0.0
    for dense,lexical,gold in rankings:
        ranking=fused(dense,lexical,alpha,limit=10)
        relevant=[i+1 for i,key in enumerate(ranking) if key in gold]
        top1+=bool(relevant and relevant[0]==1)
        top3+=bool(relevant and relevant[0]<=3)
        top5+=bool(relevant and relevant[0]<=5)
        mrr+=1/relevant[0] if relevant and relevant[0]<=5 else 0
        recall+=len(relevant)/max(1,len(gold))
        dcg=sum(1/math.log2(rank+1) for rank in relevant)
        ideal=sum(1/math.log2(rank+1) for rank in range(1,min(len(gold),10)+1))
        ndcg+=dcg/ideal if ideal else 0
    n=max(1,len(rankings))
    return {"queries":len(rankings),"top1":top1/n,"top3":top3/n,"top5":top5/n,
            "mrr_at5":mrr/n,"recall_at10":recall/n,"ndcg_at10":ndcg/n}


def bm25_rankings(rows, examples):
    bm25=BM25(rows)
    empty=torch.zeros((len(rows),1))
    return ranks_for(rows,examples,empty,bm25,torch.zeros((len(examples),1)))


def bm25_latency(rows,examples):
    engine=BM25(rows); samples=[]
    for example in examples:
        start=perf_counter()
        engine.search(example["query"],cohort=example["cohort"],subject=example["subject"],limit=10000)
        samples.append((perf_counter()-start)*1000)
    ordered=sorted(samples)
    return {"p50_ms":median(samples),"p95_ms":ordered[int(.95*(len(ordered)-1))]}


def challenge_eval(rows, cases, doc_vectors, encode_query, engine, device, abstention_threshold=None):
    details=[]; ranks=[]; unknown=[]
    for case in cases:
        eligible=[i for i,row in enumerate(rows) if row["cohort"]=="ib-2026-1kurs"
                  and row["subject"]==case["subject"] and row["layer"]==case["layer"]]
        gold={rows[i]["id"] for i in eligible if rows[i]["source"].endswith("/"+case.get("source",""))
              and case.get("anchor","") in rows[i]["text"]} if not case.get("absent") else set()
        if not case.get("absent") and not gold:
            details.append({"query":case["q"],"status":"missing_gold_anchor"}); continue
        sync(device); start=perf_counter(); qv=encode_query(case["q"]); sync(device)
        dense_order=sorted(eligible,key=lambda i:(-float(doc_vectors[i]@qv),rows[i]["id"]))
        dense=[rows[i]["id"] for i in dense_order]
        lex=[hit.passage["id"] for hit in engine.search(case["q"],cohort="ib-2026-1kurs",
             subject=case["subject"],layers={case["layer"]},limit=max(1,len(eligible)))]
        if case.get("absent"):
            top_score=float(doc_vectors[dense_order[0]]@qv) if dense_order else None
            unknown.append({"query":case["q"],"bm25_returned_any":bool(lex),
                            "dense_top1_cosine":top_score,
                            "dense_abstained":(top_score is None or (abstention_threshold is not None and top_score<abstention_threshold)),
                            "dense_top1_source":rows[dense_order[0]]["source"] if dense_order else None})
            continue
        rank=next((j for j,key in enumerate(dense,1) if key in gold),None)
        ranks.append(rank)
        top_score=float(doc_vectors[dense_order[0]]@qv) if dense_order else None
        details.append({"query":case["q"],"gold_count":len(gold),"dense_rank":rank,
                        "dense_top1_cosine":top_score,
                        "dense_abstained":(top_score is None or (abstention_threshold is not None and top_score<abstention_threshold)),
                        "bm25_top1_source":next((r["source"] for r in rows if r["id"]==lex[0]),None) if lex else None,
                        "dense_top1_source":rows[dense_order[0]]["source"] if dense_order else None})
    n=max(1,len(ranks))
    metrics={"answerable":len(ranks),"top1":sum(r==1 for r in ranks)/n,
             "top5":sum(r is not None and r<=5 for r in ranks)/n,
             "mrr_at5":sum(1/r for r in ranks if r and r<=5)/n}
    return {"metrics":metrics,"unanswerable_cases":unknown,"details":details}


def calibrate_threshold(positive,negative):
    """Select a cosine cutoff using balanced accuracy on Validation only."""
    candidates=sorted(set([min(positive+negative)-1e-6,*positive,*negative,max(positive+negative)+1e-6]))
    best=None
    for threshold in candidates:
        sensitivity=sum(score>=threshold for score in positive)/max(1,len(positive))
        specificity=sum(score<threshold for score in negative)/max(1,len(negative))
        item={"threshold":threshold,"balanced_accuracy":(sensitivity+specificity)/2,
              "answerable_recall":sensitivity,"unanswerable_recall":specificity}
        if best is None or (item["balanced_accuracy"],item["unanswerable_recall"],-threshold)>(best["balanced_accuracy"],best["unanswerable_recall"],-best["threshold"]): best=item
    return {**best,"validation_answerable":len(positive),"validation_unanswerable":len(negative)}


def validation_score_lists(rows,examples,query_vectors,doc_vectors,negative_queries,negative_encoder,negative_docs):
    scopes={}
    for i,row in enumerate(rows): scopes.setdefault((row["cohort"],row["subject"]),[]).append(i)
    positive=[]
    for i,example in enumerate(examples):
        candidates=scopes[(example["cohort"],example["subject"])]
        positive.append(float((doc_vectors[candidates]@query_vectors[i]).max()))
    negative=[]
    for query in negative_queries:
        q=negative_encoder(query)
        negative.append(float((negative_docs@q).max()))
    return positive,negative


def timed_query_vectors(encode, examples, device):
    vectors=[]; timings=[]
    for example in examples:
        sync(device); start=perf_counter()
        vec=encode(example["query"])
        sync(device); timings.append((perf_counter()-start)*1000)
        vectors.append(vec)
    return torch.stack(vectors),timings


def summarize_model(rows, examples, docs, encode, bm25, device, model_bytes):
    query_vecs,latencies=timed_query_vectors(encode,examples,device)
    ranks=ranks_for(rows,examples,docs,bm25,query_vecs)
    result=score_ranks(ranks,0)
    result.update({"query_latency_p50_ms":median(latencies),"query_latency_p95_ms":sorted(latencies)[int(.95*(len(latencies)-1))],
                   "model_bytes":model_bytes,"_ranks":ranks})
    return result


def run(args):
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; refusing CPU fallback")
    base=Path(args.directory); dataset=base/"dataset"; trained=base/"trained"
    tokenizer=BPETokenizer.load(base/"tokenizer.json")
    vrows,vexamples=read_split(dataset,"validation")
    trows,texamples=read_split(dataset,"test")
    # BM25F ranking is held constant for all systems and used to choose alpha.
    val_bm=BM25(vrows); test_bm=BM25(trows)
    bm_val=bm25_rankings(vrows,vexamples); bm_test=bm25_rankings(trows,texamples)
    v2,steps=load_checkpoint(trained/"encoder-best.pt",base/"tokenizer.json",map_location=device)
    v2v=university_docs(v2,tokenizer,vrows,device); v2t=university_docs(v2,tokenizer,trows,device)
    v2_size=(trained/"encoder-best.pt").stat().st_size+(base/"tokenizer.json").stat().st_size
    val_dense=summarize_model(vrows,vexamples,v2v,lambda q:university_query(v2,tokenizer,q,device),val_bm,device,v2_size)
    test_dense=summarize_model(trows,texamples,v2t,lambda q:university_query(v2,tokenizer,q,device),test_bm,device,v2_size)
    alpha=max((0,.25,.5,.75,1),key=lambda a:(score_ranks(val_dense["_ranks"],a)["mrr_at5"],score_ranks(val_dense["_ranks"],a)["top1"],a))
    alpha_scores={str(a):score_ranks(val_dense["_ranks"],a) for a in (0,.25,.5,.75,1)}
    train_rows,_=read_split(dataset,"train")
    calibration_rows=train_rows+vrows
    calibration_docs=university_docs(v2,tokenizer,calibration_rows,device)
    val_qv,_=timed_query_vectors(lambda q:university_query(v2,tokenizer,q,device),vexamples,device)
    no_answer=[item["q"] for item in json.loads(Path("ai/evaluation/unanswerable_validation.json").read_text(encoding="utf-8"))]
    v2_positive,v2_negative=validation_score_lists(vrows,vexamples,val_qv,v2v,no_answer,
         lambda q:university_query(v2,tokenizer,q,device),calibration_docs)
    v2_threshold=calibrate_threshold(v2_positive,v2_negative)
    hybrid={"validation":score_ranks(val_dense["_ranks"],alpha),"test":score_ranks(test_dense["_ranks"],alpha),"alpha_lexical":alpha}

    # Compare against the E5 model used as teacher, reloaded from its pinned local cache.
    e5tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,local_files_only=True)
    e5=AutoModel.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,use_safetensors=True,
                                 local_files_only=True).to(device).eval()
    e5v=e5_encode(e5,e5tok,["passage: "+r["encoder_text"] for r in vrows],device,"",32)
    e5t=e5_encode(e5,e5tok,["passage: "+r["encoder_text"] for r in trows],device,"",32)
    def e5_query(q): return e5_encode(e5,e5tok,["query: "+q],device,"",1)[0]
    e5_size=sum(p.numel()*p.element_size() for p in e5.parameters())
    e5_val=summarize_model(vrows,vexamples,e5v,e5_query,val_bm,device,e5_size)
    e5_test=summarize_model(trows,texamples,e5t,e5_query,test_bm,device,e5_size)
    e5_alpha=max((0,.25,.5,.75,1),key=lambda a:(score_ranks(e5_val["_ranks"],a)["mrr_at5"],score_ranks(e5_val["_ranks"],a)["top1"],a))
    calibration_e5=e5_encode(e5,e5tok,["passage: "+r["encoder_text"] for r in calibration_rows],device,"",32)
    e5_val_qv,_=timed_query_vectors(e5_query,vexamples,device)
    e5_positive,e5_negative=validation_score_lists(vrows,vexamples,e5_val_qv,e5v,no_answer,e5_query,calibration_e5)
    e5_threshold=calibrate_threshold(e5_positive,e5_negative)

    full_rows=[]
    for split in ("train","validation","test"):
        split_rows,_=read_split(dataset,split)
        full_rows.extend(split_rows)
    challenge_cases=json.loads(Path("ai/evaluation/questions.json").read_text(encoding="utf-8"))
    full_bm=BM25(full_rows)
    full_v2=university_docs(v2,tokenizer,full_rows,device)
    v2_challenge=challenge_eval(full_rows,challenge_cases,full_v2,
        lambda q:university_query(v2,tokenizer,q,device),full_bm,device,v2_threshold["threshold"])
    full_e5=e5_encode(e5,e5tok,["passage: "+r["encoder_text"] for r in full_rows],device,"",32)
    e5_challenge=challenge_eval(full_rows,challenge_cases,full_e5,e5_query,full_bm,device,e5_threshold["threshold"])
    bm_challenge_details=[]; bm_ranks=[]; bm_unknown=[]
    for case in challenge_cases:
        hits=full_bm.search(case["q"],subject=case["subject"],layers={case["layer"]},limit=5)
        if case.get("absent"):
            bm_unknown.append({"query":case["q"],"abstained":not hits}); continue
        gold={r["id"] for r in full_rows if r["cohort"]=="ib-2026-1kurs"
              and r["subject"]==case["subject"] and r["layer"]==case["layer"]
              and r["source"].endswith("/"+case.get("source","")) and case.get("anchor","") in r["text"]}
        if not gold: continue
        rank=next((i for i,h in enumerate(hits,1) if h.passage["id"] in gold),None); bm_ranks.append(rank)
        bm_challenge_details.append({"query":case["q"],"rank":rank,"top_source":hits[0].passage["source"] if hits else None})
    bm_challenge={"metrics":{"answerable":len(bm_ranks),"top1":sum(r==1 for r in bm_ranks)/max(1,len(bm_ranks)),
        "top5":sum(r is not None and r<=5 for r in bm_ranks)/max(1,len(bm_ranks)),
        "mrr_at5":sum(1/r for r in bm_ranks if r and r<=5)/max(1,len(bm_ranks))},
        "unanswerable_cases":bm_unknown,"details":bm_challenge_details}

    v1_path=Path(args.v1_checkpoint) if args.v1_checkpoint else Path("ai/generated/trained/encoder-best.pt")
    v1={"status":"unavailable_no_checkpoint","expected_path":str(v1_path)}
    if v1_path.exists():
        v1model,_=load_checkpoint(v1_path,base/"tokenizer.json",map_location=device)
        v1docs=university_docs(v1model,tokenizer,trows,device)
        v1=summarize_model(trows,texamples,v1docs,lambda q:university_query(v1model,tokenizer,q,device),test_bm,device,v1_path.stat().st_size)
        v1.pop("_ranks",None)

    bm25_bytes=sum((base.parent/name).stat().st_size for name in ("passages.jsonl","manifest.json"))
    report={"status":"completed","device":str(device),"university_v02_checkpoint":str(trained/"encoder-best.pt"),
        "university_v02_best_steps":steps,"teacher":{"model":MODEL,"revision":REVISION},
        "split":{"validation_queries":len(vexamples),"test_queries":len(texamples),"test_passages":len(trows)},
        "selected_alpha_v02":alpha,"alpha_selection_validation":alpha_scores,
        "test":{"BM25F":score_ranks(bm_test,1),"University_Encoder_v0.2":{k:v for k,v in test_dense.items() if k!="_ranks"},
                "Hybrid_BM25F_Encoder_v0.2":hybrid["test"],"Multilingual_E5_Small":{k:v for k,v in e5_test.items() if k!="_ranks"},
                "University_Encoder_v0.1":v1},
        "validation":{"encoder_v02":{k:v for k,v in val_dense.items() if k!="_ranks"},
                      "e5":{k:v for k,v in e5_val.items() if k!="_ranks"},
                      "e5_selected_alpha":e5_alpha},
        "bm25f_latency_test":bm25_latency(trows,texamples),"bm25f_index_bytes":bm25_bytes,
        "challenge_queries":{"cases_file":"ai/evaluation/questions.json","BM25F":bm_challenge,
                             "University_Encoder_v0.2":v2_challenge,"Multilingual_E5_Small":e5_challenge},
        "abstention_threshold":{"selected_on_validation":{"University_Encoder_v0.2":v2_threshold,
               "Multilingual_E5_Small":e5_threshold},"negative_queries_file":"ai/evaluation/unanswerable_validation.json",
               "threshold_metric":"balanced_accuracy"},
        "notes":["The v0.1 checkpoint was not present; no weights or metrics were fabricated.",
                 "Test was evaluated after alpha selection on Validation.",
                 "Latency measures per-query encoding only; corpus embeddings are precomputed."]}
    out=trained/"evaluation-report.json"; tmp=out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");tmp.replace(out)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",default="ai/generated/full")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--v1-checkpoint")
    run(parser.parse_args())
