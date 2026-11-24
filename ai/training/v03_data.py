"""Prepare v0.3 query families and a train-only passage collection."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from ai.question_bank.typos import choices, norm
from ai.search.bm25 import BM25
from ai.training.dataset import normalize


def load_rows(root, name):
    return [json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scope(row):
    return row["cohort"], row["subject"]


def category(row):
    if row["origin"] != "repository":
        return "manual" if row["label"] == "authored" else "automatic"
    source = (row.get("question_source") or {}).get("source", "")
    if any(section in source for section in ("/teacher_questions/", "/exam/", "/tests/", "/quick_cr_prep/")):
        return "exam_or_teacher_source"
    return "repository_self_check"


def valid_variant(parent, row):
    states = {parent["query"]}
    for edit in row.get("edits", ()):
        states = {text[:match.start()] + edit["after"] + text[match.end():]
                  for text in states for match in choices(text) if match[0] == edit["before"]}
    return row["query"] in states and row.get("clean_query") == parent["query"]


def shingles(text):
    text = " ".join(re.findall(r"\w+", norm(text)))
    return {text[i:i+4] for i in range(len(text)-3)}


def near(text, candidates, threshold=.86):
    key = norm(text)
    a = shingles(key)
    return any(key == other or (
        len(a) >= 12 and len(b) >= 12 and
        min(len(a),len(b)) / max(len(a),len(b)) >= threshold and
        len(a & b) / len(a | b) >= threshold
    ) for other, b in candidates)


@dataclass
class Bank:
    rows: dict
    examples: dict
    families: dict
    train_docs: list
    train_examples: list
    fingerprint: str
    audit: dict


def prepare(root, legacy, output=None):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if digest(root / name) != info["sha256"]:
            raise ValueError(f"dataset SHA-256 mismatch: {name}")
    rows = {r["id"]:r for r in load_rows(root, "passages.jsonl")}
    data = {s:load_rows(root, f"{s}-examples.jsonl") for s in ("train","validation","test")}
    if not legacy.is_file():
        raise FileNotFoundError(f"legacy holdout required: {legacy}")
    legacy_examples = load_rows(legacy.parent, legacy.name)
    legacy_docs_path = legacy.with_name("test-passages.jsonl")
    legacy_docs = load_rows(legacy_docs_path.parent, legacy_docs_path.name)
    manual_path = Path(__file__).resolve().parents[1] / "evaluation/paraphrases_test.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    held_questions = [r["query"] for s in ("validation","test") for r in data[s] if r["origin"]!="typo"]
    held_questions += [r["query"] for r in legacy_examples] + [r["q"] for r in manual]
    held = [(norm(q),shingles(q)) for q in held_questions]
    held_docs = {normalize(r["text"]) for r in [*rows.values(), *legacy_docs] if r["split"]!="train"}
    docs = [r for r in rows.values() if r["split"]=="train" and normalize(r["text"]) not in held_docs]
    doc_ids = {r["id"] for r in docs}
    parents = {r["id"]:r for split in data.values() for r in split if r["origin"]!="typo"}
    families = defaultdict(list)
    bad_variants = 0
    seen_ids = set()
    for split, examples in data.items():
        for e in examples:
            if e["id"] in seen_ids or e["split"] != split or not e["positive_ids"]:
                raise ValueError(f"bad query ID/split/positives: {e['id']}")
            seen_ids.add(e["id"])
            if any(rows[key]["split"] != split or scope(rows[key]) != scope(e) for key in e["positive_ids"]):
                raise ValueError(f"positive crosses scope: {e['id']}")
            if e["origin"] == "typo":
                parent = parents[e["parent_id"]]
                if parent["split"] != split or e["positive_ids"] != parent["positive_ids"]:
                    raise ValueError(f"typo crosses family: {e['id']}")
                if valid_variant(parent,e):
                    families[parent["id"]].append(e)
                else:
                    bad_variants += 1
    accepted = []
    excluded = Counter()
    for e in data["train"]:
        if e["origin"]=="typo":
            continue
        if not set(e["positive_ids"]) <= doc_ids:
            excluded["shared_passage"] += 1
            continue
        if near(e["query"],held):
            excluded["near_heldout_question"] += 1
            continue
        accepted.append(e)

    audit = {"manifest_sha256":digest(root/"manifest.json"), "legacy_sha256":digest(legacy),
             "legacy_passages_sha256":digest(legacy_docs_path),"manual_sha256":digest(manual_path),
             "train_groups":len(accepted),"train_passages":len(docs),
             "excluded_train_groups":dict(excluded),"invalid_typo_edits_excluded_from_training":bad_variants,
             "train_by_origin":dict(Counter(category(e) for e in accepted))}
    if output:
        output.mkdir(parents=True,exist_ok=True)
        index = BM25(list(rows.values()))
        review = []
        for e in load_rows(root,"review.jsonl"):
            hits = index.search(e["query"],cohort=e["cohort"],subject=e["subject"],limit=3)
            review.append({"id":e["id"],"query":e["query"],"status":"unverified",
                           "candidates":[{"id":hit.passage["id"],"score":hit.score,"source":hit.passage["source"],
                                          "line_start":hit.passage["line_start"],"line_end":hit.passage["line_end"]}
                                         for hit in hits]})
        audit["review"] = {"queries":len(review),"with_candidates":sum(bool(r["candidates"]) for r in review),
                           "promoted":0}
        write_json(output/"review-candidates.json",review)
        write_json(output/"dataset-audit.json",audit)
    fingerprint = hashlib.sha256(json.dumps({**{k:v for k,v in audit.items() if k!="review"},
                                             "ids":[e["id"] for e in accepted]},sort_keys=True).encode()).hexdigest()
    return Bank(rows,data,families,docs,accepted,fingerprint,audit)
