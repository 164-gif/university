"""Usage: python -m ai.search.cli 'Что такое ускорение?' --subject physics"""
import argparse
import json
from pathlib import Path

from ai.indexing.chunker import DEFAULT_COHORT
from ai.search.bm25 import BM25

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question")
    p.add_argument("--index", type=Path, default=Path("ai/generated"))
    p.add_argument("--subject")
    p.add_argument("--cohort", default=DEFAULT_COHORT, help="use 'all' to search all cohorts")
    p.add_argument("--layer", action="append")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--no-fuzzy", action="store_true")
    args = p.parse_args()
    if not 1 <= args.limit <= 50:
        p.error("--limit must be between 1 and 50")
    hits = BM25.load(args.index).search(args.question,
        subject=args.subject, cohort=None if args.cohort=="all" else args.cohort,
        layers=set(args.layer) if args.layer else None, limit=args.limit, fuzzy=not args.no_fuzzy)
    print(json.dumps({"status":"candidates" if hits else "not_found",
        "results":[{"score":round(h.score,6),"coverage":h.coverage,
                    "matched":h.matched,"corrections":h.corrected,
                    "passage":h.passage} for h in hits]},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
