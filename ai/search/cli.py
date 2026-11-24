"""Usage: python -m ai.search.cli 'Что такое ускорение?' --subject physics"""
import argparse
import json
from pathlib import Path

from ai.indexing.chunker import DEFAULT_COHORT
from ai.search.bm25 import BM25
from ai.search.rerank import rerank


def hybrid_search(args, scope):
    from hashlib import sha256

    import torch

    from ai.model.checkpoint import load_checkpoint
    from ai.tokenizer.bpe import BPETokenizer
    from ai.training.retrieval_eval import encode_many

    torch.set_num_threads(4)
    dataset = args.model_directory / "dataset"
    rows = []
    for split in ("train", "validation", "test"):
        with (dataset / f"{split}-passages.jsonl").open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    selected = json.loads(args.reranking_report.read_text(encoding="utf-8"))["selected"]
    weight = selected["weight"]
    hits = BM25(rows).search(args.question, limit=10 if weight else args.limit, **scope)
    if not hits or not weight:
        return [(hit, None) for hit in hits[:args.limit]]
    checkpoint = Path(selected["checkpoint"])
    if sha256(checkpoint.read_bytes()).hexdigest() != selected["checkpoint_sha256"]:
        raise ValueError("reranker checkpoint does not match the evaluation report")
    tokenizer_path = args.model_directory / "tokenizer.json"
    model, _ = load_checkpoint(checkpoint, tokenizer_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    if len(tokenizer.encode(args.question)) > model.config.max_length:
        return [(hit, None) for hit in hits[:args.limit]]
    vectors = encode_many(model, tokenizer, [args.question] + [hit.passage["encoder_text"] for hit in hits])
    similarities = {hit.passage["id"]: score for hit, score in zip(hits, (vectors[1:] @ vectors[0]).tolist())}
    return rerank(hits, similarities, weight)[:args.limit]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question")
    p.add_argument("--index", type=Path, default=Path("ai/generated"))
    p.add_argument("--subject")
    p.add_argument("--cohort", default=DEFAULT_COHORT, help="use 'all' to search all cohorts")
    p.add_argument("--layer", action="append")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--no-fuzzy", action="store_true")
    p.add_argument("--reranking-report", type=Path, help="use the checkpoint and weight selected in a reranking report")
    p.add_argument("--model-directory", type=Path, default=Path("ai/generated/full"))
    args = p.parse_args()
    if not 1 <= args.limit <= 50:
        p.error("--limit must be between 1 and 50")
    if args.reranking_report and args.limit > 10:
        p.error("--reranking-report supports at most 10 hits")
    scope = {"subject": args.subject, "cohort": None if args.cohort=="all" else args.cohort,
             "layers": set(args.layer) if args.layer else None, "fuzzy": not args.no_fuzzy}
    if args.reranking_report:
        hits = hybrid_search(args, scope)
    else:
        hits = [(hit, None) for hit in BM25.load(args.index).search(args.question, limit=args.limit, **scope)]
    print(json.dumps({"status":"candidates" if hits else "not_found",
        "results":[{"score":round(h.score,6),"coverage":h.coverage,
                    "matched":h.matched,"corrections":h.corrected,
                    "passage":h.passage,
                    **({"ranking_score":round(score,6)} if score is not None else {})}
                   for h,score in hits]},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
