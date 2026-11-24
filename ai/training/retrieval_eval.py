"""Compare BM25F, dense retrieval and reciprocal rank fusion on the same split."""
from collections import defaultdict
from statistics import mean

import torch

from ai.model.inputs import collate_texts
from ai.search.bm25 import BM25

def encode_many(model, tokenizer, texts, *, batch_size=16, device="cpu"):
    if not texts:
        return torch.empty((0,model.config.embedding_size))
    order=sorted(range(len(texts)),key=lambda i:len(tokenizer.encode(texts[i])))
    out=torch.empty((len(texts),model.config.embedding_size))
    was_training=model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(order),batch_size):
            idx=order[start:start+batch_size]
            batch=collate_texts(tokenizer,[texts[i] for i in idx],model.config,device=device)
            out[idx]=model(**batch).cpu()
    model.train(was_training)
    return out

def rrf(lexical,dense,alpha,k=20):
    """alpha is lexical weight; absent ranks contribute nothing."""
    if alpha==1:
        return list(lexical)
    if alpha==0:
        return list(dense)
    scores=defaultdict(float)
    for i,key in enumerate(lexical,1):
        scores[key]+=alpha/(k+i)
    for i,key in enumerate(dense,1):
        scores[key]+=(1-alpha)/(k+i)
    return sorted(scores,key=lambda key:(-scores[key],key))

def metrics(ranks):
    n=len(ranks)
    if not n:
        raise ValueError("empty evaluation")
    return {"queries":n,"top1":sum(r==1 for r in ranks)/n,
            "top5":sum(r is not None and r<=5 for r in ranks)/n,
            "mrr_at5":mean(1/r if r and r<=5 else 0 for r in ranks)}

def evaluate(model,tokenizer,rows,examples,*,device="cpu",batch_size=16,alphas=(0,.25,.5,.75,1)):
    if not rows or not examples:
        raise ValueError("empty split")
    lex=BM25(rows)
    doc=encode_many(model,tokenizer,[r["encoder_text"] for r in rows],batch_size=batch_size,device=device)
    query=encode_many(model,tokenizer,[e["query"] for e in examples],batch_size=batch_size,device=device)
    scope=defaultdict(list)
    for i,r in enumerate(rows):
        scope[(r["cohort"],r["subject"])].append(i)
    ranks={float(a):[] for a in alphas}
    for i,e in enumerate(examples):
        indices=scope[(e["cohort"],e["subject"])]
        if not indices:
            raise ValueError("missing evaluation scope")
        sim=(doc[indices]@query[i]).tolist()
        dense=[rows[j]["id"] for j,_ in sorted(zip(indices,sim),key=lambda x:(-x[1],rows[x[0]]["id"]))]
        lexical=[h.passage["id"] for h in lex.search(e["query"],cohort=e["cohort"],subject=e["subject"],
                                                     limit=len(indices))]
        gold=set(e["positive_ids"])
        for a in ranks:
            ranking=rrf(lexical,dense,a)
            ranks[a].append(next((j for j,key in enumerate(ranking,1) if key in gold),None))
    return {str(a):metrics(r) for a,r in ranks.items()}

def choose_alpha(report):
    # Prefer lexical-heavy fusion on ties, so random neural noise is not rewarded.
    return float(max(report,key=lambda a:(report[a]["mrr_at5"],report[a]["top1"],float(a))))
