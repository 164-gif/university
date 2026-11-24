"""Build versioned dataset artifacts from verified index and BPE."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ai.indexing.chunker import digest, SCHEMA_VERSION
from ai.tokenizer.bpe import BPETokenizer
from ai.training.dataset import VERSION, HOLDOUT, build_rows, validate

def run(index, output, *, max_length=512, negatives=3):
    index=Path(index)
    output=Path(output)
    manifest=json.loads((index/"manifest.json").read_text(encoding="utf-8"))
    raw=(index/"passages.jsonl").read_bytes().decode("utf-8")
    if manifest["schema_version"]!=SCHEMA_VERSION or digest(raw)!=manifest["index_sha256"]:
        raise ValueError("index integrity mismatch")
    rows=[json.loads(line) for line in raw.splitlines() if line]
    if len(rows)!=manifest["passages"]:
        raise ValueError("index count mismatch")
    tok_path=index/"tokenizer.json"
    tok_hash=hashlib.sha256(tok_path.read_bytes()).hexdigest()
    tok_report=json.loads((index/"tokenizer-report.json").read_text(encoding="utf-8"))
    if tok_report["corpus_sha256"]!=manifest["index_sha256"] or tok_report["tokenizer_sha256"]!=tok_hash:
        raise ValueError("tokenizer does not match corpus")
    tokenizer=BPETokenizer.load(tok_path)
    result,stats=build_rows(rows,tokenizer,max_length=max_length,negatives=negatives)
    validate(result,tokenizer,max_length)
    output.mkdir(parents=True,exist_ok=True)
    report={"schema_version":VERSION,"corpus_sha256":manifest["index_sha256"],
            "tokenizer_sha256":tok_hash,"max_length":max_length,
            "negative_limit":negatives,"holdout_subjects":HOLDOUT,
            "excluded":stats,"splits":{},"files":{}}
    for s,(passages,examples) in result.items():
        for kind,data in (("passages",passages),("examples",examples)):
            name=f"{s}-{kind}.jsonl"
            body="".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in data)
            tmp=output/(name+".tmp")
            tmp.write_bytes(body.encode("utf-8"))
            tmp.replace(output/name)
            report["files"][name]=digest(body)
        report["splits"][s]={"passages":len(passages),"examples":len(examples),
            "positive_pairs":sum(len(e["positive_ids"]) for e in examples),
            "negative_candidates":sum(len(e["negative_candidates"]) for e in examples),
            "examples_without_negatives":sum(not e["negative_candidates"] for e in examples),
            "subjects":dict(sorted(Counter(r["subject"] for r in passages).items())),
            "query_kinds":dict(sorted(Counter(k for e in examples for k in e["query_kinds"]).items()))}
    (output/"dataset-manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=Path,default=Path("ai/generated"))
    p.add_argument("--output",type=Path,default=Path("ai/generated/dataset"))
    p.add_argument("--max-length",type=int,default=512)
    p.add_argument("--negatives",type=int,default=3)
    args=p.parse_args()
    report=run(args.index,args.output,max_length=args.max_length,negatives=args.negatives)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if any(not report["splits"][s]["examples"] for s in ("train","validation","test")):
        raise SystemExit("empty dataset split")

if __name__=="__main__":
    main()
