"""Validate the checked-in v0.3 question bank without changing source data."""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate(root):
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    expected={"existing.jsonl":1330,"generated.jsonl":3000,"typos.jsonl":16440,
              "review.jsonl":1033,"passages.jsonl":2249,"train-examples.jsonl":15291,
              "validation-examples.jsonl":2280,"test-examples.jsonl":2166,
              "generated-extra-2000.jsonl":2000,"typos-extra.jsonl":10000}
    files={name:read(root/name) for name in expected}
    errors=[]
    for name,count in expected.items():
        raw=(root/name).read_bytes()
        if len(files[name])!=count: errors.append(f"{name}: count {len(files[name])}, expected {count}")
        info=manifest["files"][name]
        if len(files[name])!=info["rows"] or hashlib.sha256(raw).hexdigest()!=info["sha256"]:
            errors.append(f"{name}: manifest mismatch")
    passages={r["id"]:r for r in files["passages.jsonl"]}
    if len(passages)!=len(files["passages.jsonl"]): errors.append("duplicate passage IDs")
    sources={}
    source_commit=manifest["source_commit"]
    for row in passages.values():
        path=row["source"]
        if path not in sources:
            blob=subprocess.run(["git","show",f"{source_commit}:{path}"],check=True,capture_output=True).stdout
            sources[path]=(hashlib.sha256(blob).hexdigest(),blob.decode("utf-8").splitlines(keepends=True))
        source_sha,lines=sources[path]
        start,end=row["line_start"],row["line_end"]
        text="".join(lines[start-1:end])
        if source_sha!=row["source_sha256"] or text!=row["text"] or hashlib.sha256(text.encode()).hexdigest()!=row["text_sha256"]:
            errors.append(f"passage source or line mismatch: {row['id']}")
    splits={name.removesuffix("-examples.jsonl"):rows for name,rows in files.items() if name.endswith("-examples.jsonl")}
    qids=set(); parent_split={}
    for split,rows in splits.items():
        for row in rows:
            if row["id"] in qids: errors.append(f"duplicate query ID: {row['id']}")
            qids.add(row["id"])
            if row["split"]!=split: errors.append(f"split mismatch: {row['id']}")
            if row["label"]!="review":
                for pid in row["positive_ids"]:
                    if pid not in passages: errors.append(f"missing positive passage {pid}")
                    elif passages[pid]["subject"]!=row["subject"] or passages[pid]["cohort"]!=row["cohort"]:
                        errors.append(f"subject/cohort mismatch {row['id']} -> {pid}")
            by_id={s["id"]:s for s in row.get("positive_sources",[])}
            if set(by_id)!=set(row["positive_ids"]):
                errors.append(f"positive metadata differs: {row['id']}")
            for key,reference in by_id.items():
                if key in passages and any(reference.get(field)!=passages[key][field]
                                            for field in ("source","line_start","line_end","text_sha256")):
                    errors.append(f"positive source mismatch: {row['id']} -> {key}")
            parent_split[row["id"]]=split
    typo_counts=Counter(); typo_bad=0
    for r in files["typos.jsonl"]:
        if r.get("parent_id") not in parent_split or r.get("split")!=parent_split.get(r.get("parent_id")):
            typo_bad+=1
        typo_counts[r.get("split")]+=1
    for extra, combined in (("generated-extra-2000.jsonl","generated.jsonl"),
                            ("typos-extra.jsonl","typos.jsonl")):
        parents={r["id"]:r for r in files[combined]}
        if len(parents)!=len(files[combined]) or any(parents.get(r["id"])!=r for r in files[extra]):
            errors.append(f"{extra}: inconsistent with combined file")
    # Exact normalized duplicate scan across assigned splits, retaining parent grouping.
    normalized={}
    for split,rows in splits.items():
        for r in rows:
            key=" ".join(r["query"].casefold().replace("ё","е").split())
            normalized.setdefault(key,set()).add(split)
    cross_split=sum(len(v)>1 for v in normalized.values())
    report={"errors":errors[:100],"counts":{n:len(v) for n,v in files.items()},
            "passages":len(passages),"source_documents":len(sources),"typos_by_split":dict(typo_counts),
            "typo_parent_split_mismatches":typo_bad,"cross_split_exact_question_duplicates":cross_split,
            "source_commit":manifest.get("source_commit")}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors or typo_bad or cross_split: raise SystemExit(1)
    return report


if __name__=="__main__":
    validate(Path("ai/data/v03"))
