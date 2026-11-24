"""Scope-aware ranking and metrics for the v0.3 experiments."""
from collections import defaultdict
from statistics import mean
from time import perf_counter
import math
import re

import torch
from ai.search.bm25 import BM25
from ai.training.retrieval_eval import encode_many
from ai.training.v03_data import scope


class Ranking:
    def __init__(self, rows, examples):
        self.rows = rows
        self.examples = examples
        self.gold = [set(row["positive_ids"]) for row in examples]
        pools = defaultdict(list)
        for index, row in enumerate(rows):
            pools[scope(row)].append(index)
        self.pools = [pools[scope(row)] for row in examples]
        self.ids = [row["id"] for row in rows]
        self.indices = {key:i for i,key in enumerate(self.ids)}

    def dense(self, queries, docs):
        scores = (queries.float() @ docs.float().T).cpu()
        return [sorted(pool, key=lambda j: (-value[j], self.ids[j]))
                for pool, value in zip(self.pools, scores.tolist())], scores

    def lexical(self, fuzzy):
        engine = BM25(self.rows)
        scores, times = [], []
        for row in self.examples:
            start = perf_counter()
            hits = engine.search(row["query"], cohort=row["cohort"], subject=row["subject"],
                                 limit=len(self.rows), fuzzy=fuzzy)
            times.append((perf_counter()-start)*1000)
            scores.append({self.indices[hit.passage["id"]]:hit.score for hit in hits})
        return scores, latency(times)

    def summaries(self, rankings, detailed=False):
        per_query = []
        for order, gold in zip(rankings, self.gold):
            positions = [n for n, j in enumerate(order[:30], 1) if self.ids[j] in gold]
            first = positions[0] if positions else 100
            dcg = sum(1/math.log2(n+1) for n in positions if n<=10)
            ideal = sum(1/math.log2(n+1) for n in range(1,min(10,len(gold))+1))
            per_query.append({"top1":float(first==1), "top3":float(first<=3), "top5":float(first<=5),
                              "mrr_at5":1/first if first<=5 else 0., "recall_at30":len(positions)/len(gold),
                              "ndcg_at10":dcg/ideal})
        def average(values):
            return {k:mean(r[k] for r in values) for k in values[0]} if values else {}
        families = defaultdict(list)
        subsets = defaultdict(list)
        for example, item in zip(self.examples, per_query):
            families[example.get("parent_id") or example["id"]].append(item)
            if detailed:
                edits = [e["type"] for e in example.get("edits",[])]
                subsets["typos" if edits else "clean"].append(item)
                if edits:
                    subsets["one_edit" if len(edits)==1 else "multiple_edits"].append(item)
                for edit in set(edits):
                    subsets[f"edit_{edit}"].append(item)
                if "formula" in example.get("origin","") or re.search(r"\$|\\[a-zA-Z]+",example["query"]):
                    subsets["formulas"].append(item)
                if example.get("label")=="authored" or "rephrase" in example.get("origin",""):
                    subsets["authored_or_rephrased"].append(item)
        value = {"queries":len(per_query),"groups":len(families),**average(per_query),
                 "group_mean":average([average(group) for group in families.values()])}
        if detailed:
            value["subsets"]={key:{"queries":len(rows),**average(rows)} for key,rows in subsets.items()}
        return value


def latency(values):
    values=sorted(values)
    return {f"p{n}_ms":values[min(len(values)-1,math.ceil(n*len(values)/100)-1)] for n in (50,95)}


def encode_rank(model, tokenizer, ranking, device, batch):
    docs = encode_many(model,tokenizer,[r["encoder_text"] for r in ranking.rows],device=device,batch_size=batch)
    queries = encode_many(model,tokenizer,[r["query"] for r in ranking.examples],device=device,batch_size=batch)
    return ranking.dense(queries, docs)


def fuse(ranking, lexical, dense, values, method, weight):
    output = []
    for pool, lex, semantic, scores in zip(ranking.pools, lexical, dense, values.tolist()):
        if weight == 1:
            output.append(list(lex))
            continue
        if weight == 0:
            output.append(semantic)
            continue
        if method == "rrf":
            combined={j:(1-weight)/(20+n) for n,j in enumerate(semantic,1)}
            for n,j in enumerate(lex,1):
                combined[j]+=weight/(20+n)
        else:
            top=max(lex.values(),default=1.)
            low=scores[semantic[-1]]
            spread=max(1e-8,scores[semantic[0]]-low)
            alpha=weight
            if method=="adaptive":
                first=list(lex.values())[:2]
                margin=(first[0]-first[1])/first[0] if len(first)>1 else float(bool(first))
                alpha=min(.995,weight+(1-weight)*margin) if first else 0.
            combined={j:alpha*lex.get(j,0.)/top+(1-alpha)*(scores[j]-low)/spread for j in pool}
        output.append(sorted(combined,key=lambda j:(-combined[j],ranking.ids[j])))
    return output
