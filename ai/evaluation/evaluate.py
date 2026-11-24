"""Reproducible smoke benchmark, with source+anchor labels resolved to IDs."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from statistics import median

from ai.search.bm25 import BM25

def evaluate(engine, cases):
    ranks = []
    absent = []
    details = []
    timings = []
    for case in cases:
        eligible = [r for r in engine.records if r["cohort"]=="ib-2026-1kurs"
                    and r["subject"]==case["subject"] and r["layer"]==case["layer"]]
        gold = {r["id"] for r in eligible if r["source"].endswith("/"+case.get("source",""))
                and case.get("anchor","") in r["text"]} if not case.get("absent") else set()
        if not case.get("absent") and not gold:
            raise ValueError("missing gold passage: "+case["q"])
        start = perf_counter()
        hits = engine.search(case["q"],subject=case["subject"],layers={case["layer"]},limit=5)
        timings.append((perf_counter()-start)*1000)
        rank = next((i for i,h in enumerate(hits,1) if h.passage["id"] in gold),None)
        if case.get("absent"):
            absent.append(not hits)
        else:
            ranks.append(rank)
        details.append({"q":case["q"],"rank":rank,"gold_count":len(gold),
                        "top_source":hits[0].passage["source"] if hits else None,
                        "abstained":not hits})
    n = max(1,len(ranks))
    return {"answerable":len(ranks),"top1":sum(r==1 for r in ranks)/n,
            "top5":sum(r is not None for r in ranks)/n,
            "mrr_at5":sum(1/r for r in ranks if r)/n,
            "unanswerable":len(absent),"correct_abstentions":sum(absent),
            "median_query_ms":round(median(timings),3) if timings else None,
            "details":details}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index",type=Path,default=Path("ai/generated"))
    p.add_argument("--cases",type=Path,default=Path(__file__).with_name("questions.json"))
    args = p.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(BM25.load(args.index),cases),ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
