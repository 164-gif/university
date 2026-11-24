"""Select a BM25F reranker and checkpoint using Validation, then score on Test."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from ai.model.checkpoint import load_checkpoint
from ai.search.bm25 import BM25
from ai.search.rerank import rerank
from ai.tokenizer.bpe import BPETokenizer
from ai.training.evaluate_full import read_split, score_ranks
from ai.training.manual_eval import manual_examples
from ai.training.retrieval_eval import encode_many


def query_scores(model, tokenizer, rows, examples, device):
    lexical = BM25(rows)
    doc_vectors = encode_many(model, tokenizer, [r["encoder_text"] for r in rows], device=device)
    query_vectors = encode_many(model, tokenizer, [e["query"] for e in examples], device=device)
    indices = {r["id"]: i for i, r in enumerate(rows)}
    scopes = {}
    for row in rows:
        scopes.setdefault((row["cohort"], row["subject"]), []).append(row["id"])

    for example, vector in zip(examples, query_vectors):
        cohort, subject = example["cohort"], example["subject"]
        scores = (doc_vectors @ vector).tolist()
        similarities = {key: scores[indices[key]] for key in scopes[cohort, subject]}
        hits = lexical.search(example["query"], cohort=cohort, subject=subject, limit=10)
        yield hits, similarities, set(example["positive_ids"])


def compare(scored, weight):
    rankings = []
    for hits, similarities, positives in scored:
        # Keep BM25F eligibility and use cosine differences to resolve close scores.
        ordered = hits
        if hits and weight:
            ordered = [hit for hit, _ in rerank(hits, similarities, weight)]
        rankings.append(([], [hit.passage["id"] for hit in ordered], positives))
    return score_ranks(rankings, 1)


def dense_metrics(scored):
    rankings = [(sorted(scores, key=lambda key: (-scores[key], key)), [], positives)
                for _, scores, positives in scored]
    return score_ranks(rankings)


def run(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    tokenizer_path = args.directory / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    dataset = args.directory / "dataset"
    validation = read_split(dataset, "validation")
    candidates = []
    for path in args.checkpoint:
        model, _ = load_checkpoint(path, tokenizer_path, map_location=device)
        scores = list(query_scores(model, tokenizer, *validation, device))
        candidates.append({
            "checkpoint": str(path),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dense": dense_metrics(scores),
            "reranker": [{"weight": weight, "metrics": compare(scores, weight)}
                         for weight in (0, .1, .25, .5, 1, 2)],
        })
        del model
    choices = ((item, entry["weight"], entry["metrics"])
               for item in candidates for entry in item["reranker"])
    selected, weight, metrics = max(choices, key=lambda item: (
        item[2]["mrr_at5"], item[2]["top1"], -item[1]))
    path = selected["checkpoint"]
    report = {"validation": candidates,
              "selected": {"checkpoint": path, "checkpoint_sha256": selected["checkpoint_sha256"],
                           "weight": weight, "validation": metrics}}
    if not args.validation_only:
        model, _ = load_checkpoint(path, tokenizer_path, map_location=device)
        test_rows, test_examples = read_split(dataset, "test")
        scores = list(query_scores(model, tokenizer, test_rows, test_examples, device))
        report["test"] = {"bm25f": compare(scores, 0), "dense": dense_metrics(scores),
                          "selected": compare(scores, weight)}
        cases_path = Path(__file__).parents[1] / "evaluation" / "paraphrases_test.json"
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        paraphrases, missing = manual_examples(test_rows, cases)
        report["paraphrases_test"] = {"authorship": "assistant_authored_not_human_validated",
                                      "missing_anchors": missing, "scored": len(paraphrases)}
        if paraphrases:
            scores = list(query_scores(model, tokenizer, test_rows, paraphrases, device))
            report["paraphrases_test"].update({"bm25f": compare(scores, 0),
                                               "dense": dense_metrics(scores),
                                               "selected": compare(scores, weight)})
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report["selected"], ensure_ascii=False))
    if "test" in report:
        print(json.dumps(report["test"], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("ai/generated/full"))
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-only", action="store_true")
    run(parser.parse_args())
