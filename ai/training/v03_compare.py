"""Select retrieval settings on Validation, then score held-out test sets."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import torch

from ai.model.checkpoint import load_checkpoint
from ai.model.inputs import collate_texts
from ai.question_bank.typos import ENG, RUS, choices
from ai.search.bm25 import BM25
from ai.search.lexical import tokens
from ai.training.evaluate_full import calibrate_threshold
from ai.training.retrieval_eval import encode_many
from ai.training.v03_cross import load_cross, predict, rerank
from ai.training.v03_data import load_rows, write_json
from ai.training.v03_eval import Ranking, encode_rank, fuse, latency
from ai.training.v03_experiments import V02, cached_tokenizer, device_for_cuda

LAYOUT = str.maketrans("".join(ENG),"".join(RUS))
METHODS = ("rrf","weighted","adaptive")


def correct_layout(text, lexicon):
    def replace(match):
        word = match[0]
        converted = word.lower().translate(LAYOUT)
        if set(tokens(word)) & lexicon or not (set(tokens(converted)) & lexicon):
            return word
        return converted.capitalize() if word[0].isupper() else converted
    # Leave formula spans and quoted literals as written.
    from ai.question_bank.typos import SPECIAL
    output, start = [], 0
    for match in SPECIAL.finditer(text):
        output.extend((re.sub(r"[a-zA-Z]{5,}",replace,text[start:match.start()]),match[0]))
        start = match.end()
    output.append(re.sub(r"[a-zA-Z]{5,}",replace,text[start:]))
    return "".join(output)


def lexical_with_layout(ranking):
    engine = BM25(ranking.rows)
    lexicon = set(engine.postings)
    scores, timing = [], []
    for row in ranking.examples:
        start = perf_counter()
        query = correct_layout(row["query"],lexicon)
        hits = engine.search(query,cohort=row["cohort"],subject=row["subject"],
                             limit=len(ranking.rows),fuzzy=True)
        timing.append((perf_counter()-start)*1000)
        scores.append({ranking.indices[h.passage["id"]]:h.score for h in hits})
    return scores,latency(timing)


def byte_features(training_docs, validation):
    """A train-fitted sparse byte n-gram baseline for typo robustness."""
    import numpy as np

    def grams(text):
        data = (" ".join(text.casefold().split())).encode()
        return Counter(data[i:i+n] for n in (3,4,5) for i in range(len(data)-n+1))

    frequency = Counter()
    for row in training_docs:
        frequency.update(grams(row["encoder_text"]).keys())
    vocab = {token:i for i,(token,_) in enumerate(frequency.most_common(24000))}
    idf = {token:1+np.log((len(training_docs)+1)/(frequency[token]+1)) for token in vocab}

    def features(text):
        values = {token:(1+np.log(freq))*idf[token]
                  for token,freq in grams(text).items() if token in vocab}
        norm = max(1e-12,np.sqrt(sum(value*value for value in values.values())))
        return {token:value/norm for token,value in values.items()}

    postings = defaultdict(list)
    for index,row in enumerate(validation.rows):
        for token,value in features(row["encoder_text"]).items():
            postings[token].append((index,value))
    similarity = np.zeros((len(validation.examples),len(validation.rows)),dtype=np.float32)
    for index,row in enumerate(validation.examples):
        for token,value in features(row["query"]).items():
            for column,weight in postings[token]:
                similarity[index,column] += value*weight
    order = [sorted(pool,key=lambda i:(-row[i],validation.ids[i]))
             for pool,row in zip(validation.pools,similarity)]
    return validation.summaries(order,detailed=True)


def concept_probes(model, tok, rows, device):
    cases = json.loads(Path("ai/evaluation/v03_concept_pairs.json").read_text(encoding="utf-8"))
    docs = {r["id"]:r for r in rows}
    pairs = [(case["question"],docs[case[key]]["encoder_text"])
             for case in cases for key in ("positive","negative")]
    scores = predict(model,tok,pairs,device,32).reshape(-1,2).tolist()
    return {"note":"Authored capability probes. Some source passages are in Train.",
            "accuracy":sum(pos>neg for pos,neg in scores)/len(cases),
            "cases":[{**case,"positive_score":pos,"negative_score":neg,
                      "source_splits":[docs[case[key]]["split"] for key in ("positive","negative")]}
                     for case,(pos,neg) in zip(cases,scores)]}


def build_diagnostics(examples):
    # No modifications to mathematical expressions, numbers or protected words.
    rows = []
    for ex in examples:
        if ex["origin"]=="typo":
            continue
        words = choices(ex["query"])
        selected = {m.span():m[0] for m in words}
        changes = {}
        for a,b in zip(words,words[1:]):
            if ex["query"][a.end():b.start()] == " ":
                changes["missing_space"] = ex["query"][:a.end()]+ex["query"][b.start():]
                changes["extra_space"] = ex["query"][:a.end()]+"   "+ex["query"][b.start():]
                break
        for (start,end), word in selected.items():
            bad = re.sub(r"тся$","ться",word) if word.endswith("тся") and not word.endswith("ться") else ""
            if bad and bad!=word:
                changes["orthography"] = ex["query"][:start]+bad+ex["query"][end:]
                break
        for kind, text in changes.items():
            rows.append({**ex,"id":f"{ex['id']}:{kind}","parent_id":ex["id"],
                         "origin":"typo","query":text,"edits":[{"type":kind}]})
    return rows


def evaluate_config(ranking, dense, values, lexical, settings):
    return fuse(ranking,lexical,dense,values,settings["method"],settings["weight"])


def encoder(path, device):
    tokenizer_path=path.with_name("tokenizer.json")
    model,_=load_checkpoint(path,tokenizer_path,map_location=device)
    return model,cached_tokenizer(tokenizer_path)


def choose(validation, model_scores, lexical):
    results = []
    for name,(order, scores) in model_scores.items():
        for method in METHODS:
            for weight in (.25,.5,.65,.75,.85,.92,.97,.99):
                config = {"model":name,"method":method,"weight":weight}
                result = validation.summaries(evaluate_config(validation,order,scores,lexical,config))
                results.append({**config,"metrics":result})
    return max(results,key=lambda item:item["metrics"]["group_mean"]["mrr_at5"]),results


def query_latency(model, tokenizer, ranking, device):
    samples = []
    with torch.inference_mode():
        for row in ranking.examples[::max(1,len(ranking.examples)//150)][:150]:
            start = perf_counter()
            model(**collate_texts(tokenizer,[row["query"]],model.config,device=device))
            torch.cuda.synchronize(device)
            samples.append((perf_counter()-start)*1000)
    return latency(samples)


def pipeline_latency(ranking, model, tok, cross, cross_tok, settings, selected, device):
    docs = encode_many(model,tok,[r["encoder_text"] for r in ranking.rows],device=device,batch_size=64)
    engine = BM25(ranking.rows)
    lexicon = set(engine.postings)
    times, times_cross = [], []
    with torch.inference_mode():
        for example in ranking.examples[::max(1,len(ranking.examples)//80)][:80]:
            start = perf_counter()
            query = correct_layout(example["query"],lexicon)
            hits = engine.search(query,cohort=example["cohort"],subject=example["subject"],
                                 fuzzy=True,limit=len(ranking.rows))
            indices = {ranking.indices[hit.passage["id"]]:hit.score for hit in hits}
            inputs = collate_texts(tok,[example["query"]],model.config,device=device)
            vector = model(**inputs).float().cpu()
            one = Ranking(ranking.rows,[example])
            dense, scores = one.dense(vector,docs)
            order = evaluate_config(one,dense,scores,[indices],settings)
            middle = perf_counter()
            cross_order(one,cross,cross_tok,order,selected["weight"],selected["count"],device)
            times.append((middle-start)*1000)
            times_cross.append((perf_counter()-start)*1000)
    return latency(times),latency(times_cross)


def cross_order(ranking, model, tok, order, weight, count, device):
    if weight==0:
        return order
    candidates = [row[:count] for row in order]
    ranked = rerank(model,tok,ranking,candidates,device,64)
    combined = []
    for initial, reordering, full in zip(candidates,ranked,order):
        old = {j:i for i,j in enumerate(initial)}
        new = {j:i for i,j in enumerate(reordering)}
        combined.append(sorted(initial,key=lambda j:((1-weight)*old[j]+weight*new[j],old[j]))+full[count:])
    return combined


def run(args):
    device = device_for_cuda()
    cross = load_cross(args.cross,args.cross.with_name("tokenizer.json"),device)
    cross_tok = cached_tokenizer(args.cross.with_name("tokenizer.json"))
    paths = {"v02":V02/"encoder-best.pt","a":args.a,"b":args.b}
    models = {name:encoder(path,device) for name,path in paths.items()}
    passages = load_rows(args.data,"passages.jsonl")
    val = Ranking([r for r in passages if r["split"]=="validation"],
                  load_rows(args.data,"validation-examples.jsonl"))
    val_lex,_ = lexical_with_layout(val)
    val_models = {name:encode_rank(model,tok,val,device,64) for name,(model,tok) in models.items()}
    best, grid = choose(val,val_models,val_lex)
    first = evaluate_config(val,*val_models[best["model"]],val_lex,best)
    negative_path = Path("ai/evaluation/unanswerable_validation.json")
    unknowns = [row["q"] for row in json.loads(negative_path.read_text(encoding="utf-8"))]
    dense_model, dense_tok = models[best["model"]]
    val_docs = encode_many(dense_model,dense_tok,[r["encoder_text"] for r in val.rows],device=device,batch_size=64)
    negative_vectors = encode_many(dense_model,dense_tok,unknowns,device=device,batch_size=64)
    negative_dense = (negative_vectors@val_docs.T).max(dim=1).values.tolist()
    dense_order,dense_values = val_models[best["model"]]
    positive_dense = [float(scores[order[0]]) for order,scores in zip(dense_order,dense_values)]
    lexical_index = BM25(val.rows)
    negative_lexical = [max((hit.score for hit in lexical_index.search(q,cohort=None,limit=1)),default=0.)
                        for q in unknowns]
    positive_lexical = [max(hit.values(),default=0.) for hit in val_lex]
    rejection = {"lexical":calibrate_threshold(positive_lexical,negative_lexical),
                 "dense":calibrate_threshold(positive_dense,negative_dense),
                 "sets":"answerable Validation and 12 authored unrelated requests"}
    # Score the largest candidate set once then use rank fusion on prefixes.
    candidates = [order[:20] for order in first]
    cross_val = rerank(cross,cross_tok,val,candidates,device,64)
    cross_grid = []
    for count in (10,20):
        for weight in (0.,.2,.5,1.):
            output = []
            for first_order, cross_rank in zip(first,cross_val):
                old = {j:i for i,j in enumerate(first_order[:count])}
                new = {j:i for i,j in enumerate(j for j in cross_rank if j in old)}
                output.append(sorted(old,key=lambda j:((1-weight)*old[j]+weight*new[j],old[j]))+
                              first_order[count:])
            cross_grid.append({"count":count,"weight":weight,"metrics":val.summaries(output)})
    selected_cross = max(cross_grid,key=lambda item:item["metrics"]["group_mean"]["mrr_at5"])
    val_bm25 = val.summaries([list(row) for row in val_lex])
    nominations = {"bm25_corrected":val_bm25["group_mean"]["mrr_at5"],
                   **{name:val.summaries(order)["group_mean"]["mrr_at5"]
                      for name,(order,_) in val_models.items()},
                   "hybrid":best["metrics"]["group_mean"]["mrr_at5"],
                   "hybrid_cross":selected_cross["metrics"]["group_mean"]["mrr_at5"]}
    selected = max(nominations,key=nominations.get)
    best_dense = max(val_models,key=lambda name:nominations[name])
    report = {"rejection_validation":rejection,"selection":{"search":selected,"dense_export":best_dense,"hybrid":best,"cross":selected_cross,
                           "group_mrr_at5":nominations},
              "validation":{"hybrid_grid":grid,"cross_grid":cross_grid,"bm25_corrected":val_bm25},
              "models":{name:{"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                              "checkpoint_bytes":path.stat().st_size,"parameters":models[name][0].parameter_count()}
                        for name,path in paths.items()}}
    report["models"]["cross"]={"sha256":hashlib.sha256(args.cross.read_bytes()).hexdigest(),
                               "checkpoint_bytes":args.cross.stat().st_size,"parameters":cross.parameter_count()}
    report["validation"]["byte_ngram"] = byte_features(
        [row for row in passages if row["split"]=="train"],val)
    report["concept_probes"] = concept_probes(cross,cross_tok,passages,device)
    # Selection is frozen in the persisted file before loading either held-out test.
    write_json(args.output.with_name("selection.json"),report)
    test = load_rows(args.data,"test-examples.jsonl")
    legacy_rows = load_rows(args.legacy.parent,"test-passages.jsonl")
    legacy = load_rows(args.legacy.parent,args.legacy.name)
    test_rows = [r for r in passages if r["split"]=="test"]
    splits = [("v03_synthetic",test_rows,test),("legacy",legacy_rows,legacy),
              ("spacing_and_orthography",test_rows,build_diagnostics(test))]
    for label,docs,queries in splits:
        if not queries:
            continue
        ranking = Ranking(docs,queries)
        plain, plain_ms = ranking.lexical(False)
        corrected, corrected_ms = lexical_with_layout(ranking)
        report[label] = {
            "bm25f":{**ranking.summaries([list(row) for row in plain],detailed=True),"latency":plain_ms},
            "bm25_corrected":{**ranking.summaries([list(row) for row in corrected],detailed=True),"latency":corrected_ms},
        }
        for name,(model,tok) in models.items():
            torch.cuda.reset_peak_memory_stats(device)
            order,scores = encode_rank(model,tok,ranking,device,64)
            result = ranking.summaries(order,detailed=True)
            if label=="v03_synthetic":
                result["query_encoder_latency"] = query_latency(model,tok,ranking,device)
                result["peak_gpu_mib_including_loaded_models"] = torch.cuda.max_memory_allocated(device)/2**20
            report[label][name] = result
            if name==best["model"]:
                candidate_order = evaluate_config(ranking,order,scores,corrected,best)
        report[label]["hybrid"] = ranking.summaries(candidate_order,detailed=True)
        sorted_cross = cross_order(ranking,cross,cross_tok,candidate_order,
                                   selected_cross["weight"],selected_cross["count"],device)
        report[label]["hybrid_cross"] = ranking.summaries(sorted_cross,detailed=True)
        gate = rejection["lexical"]["threshold"]
        covered = sum(max(scores.values(),default=0.)>=gate for scores in corrected)
        gated = [order if max(scores.values(),default=0.)>=gate else []
                 for scores,order in zip(corrected,sorted_cross)]
        report[label]["hybrid_cross_abstention"] = {
            **ranking.summaries(gated,detailed=True),"coverage":covered/len(queries)}
        if label=="v03_synthetic":
            hybrid_ms,cross_ms = pipeline_latency(ranking,*models[best["model"]],cross,cross_tok,
                                                  best,selected_cross,device)
            report[label]["hybrid"]["latency"] = hybrid_ms
            report[label]["hybrid_cross"]["latency"] = cross_ms
            errors = [{"id":row["id"],"query":row["query"],"gold":row["positive_ids"],
                       "first_match":ranking.rows[order[0]]["id"] if order else None}
                      for row,order in zip(queries,sorted_cross)
                      if not order or ranking.rows[order[0]]["id"] not in row["positive_ids"]]
            write_json(args.output.with_name("test-errors.json"),errors)
        write_json(args.output,report)
        print(label,{key:round(value["group_mean"]["mrr_at5"],4) for key,value in report[label].items()},flush=True)
    report["complete"] = True
    write_json(args.output,report)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("ai/data/v03"))
    parser.add_argument("--legacy",type=Path,default=Path("ai/generated/full/dataset/test-examples.jsonl"))
    for name in ("a","b","cross","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())
