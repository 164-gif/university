"""Assistant-authored source-anchored paraphrases, never used for tuning."""
import json
from pathlib import Path
from ai.training.retrieval_eval import evaluate

def evaluate_manual(model,tokenizer,rows,cases,*,device,alpha):
    examples=[]
    missing=[]
    for case in cases:
        gold=[r["id"] for r in rows if r["source"].endswith("/"+case["source"])
              and case["anchor"] in r["text"] and r["cohort"]=="ib-2026-1kurs"]
        if not gold:
            missing.append(case["q"])
            continue
        examples.append({"query":case["q"],"cohort":"ib-2026-1kurs",
                         "subject":"russian_statehood_fundamentals","positive_ids":gold})
    if not examples:
        raise ValueError("no valid manual evaluation anchors")
    return {"authorship":"assistant_authored_not_human_validated","missing":missing,
            "metrics":evaluate(model,tokenizer,rows,examples,device=device,alphas=tuple(sorted({0.,1.,alpha})))}
