"""Train BPE on the source-addressable University corpus and report compression."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

from ai.indexing.chunker import digest, SCHEMA_VERSION
from ai.tokenizer.bpe import train, split, BPETokenizer

def run(index: Path, output: Path, vocab_size=16384, min_frequency=2):
    manifest = json.loads((index/"manifest.json").read_text(encoding="utf-8"))
    raw = (index/"passages.jsonl").read_bytes().decode("utf-8")
    if manifest["schema_version"] != SCHEMA_VERSION or digest(raw) != manifest["index_sha256"]:
        raise ValueError("index integrity mismatch")
    rows = [json.loads(x) for x in raw.splitlines() if x]
    if len(rows) != manifest["passages"]:
        raise ValueError("index count mismatch")
    # Deduplicate exact training passages; do not rewrite educational sources.
    unique = {r["text_sha256"]:r["text"] for r in rows}
    texts = list(unique.values())
    if not texts:
        raise ValueError("empty corpus")
    start = perf_counter()
    tokenizer = train(texts,vocab_size=vocab_size,min_frequency=min_frequency)
    elapsed = perf_counter()-start
    output.mkdir(parents=True,exist_ok=True)
    target = output/"tokenizer.json"
    tmp = output/"tokenizer.json.tmp"
    tokenizer.save(tmp)
    tmp.replace(target)
    restored = BPETokenizer.load(target)
    total_bytes = total_chars = total_tokens = overflow = 0
    lengths = []
    for text in texts:
        ids = restored.encode(text,add_special=False)
        if restored.decode(ids) != text:
            raise ValueError("corpus roundtrip failed")
        total_bytes += len(text.encode("utf-8"))
        total_chars += len(text)
        total_tokens += len(ids)
        overflow += len(ids)>510
        lengths.append(len(ids))
    lengths.sort()
    report = {
        "corpus_sha256":manifest["index_sha256"],
        "tokenizer_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
        "requested_vocab_size":vocab_size,
        "actual_vocab_size":restored.vocab_size,
        "merges":len(restored.merges),
        "min_frequency":min_frequency,
        "unique_passages":len(texts),
        "source_passages":len(rows),
        "utf8_bytes":total_bytes,
        "unicode_chars":total_chars,
        "tokens":total_tokens,
        "bytes_per_token":round(total_bytes/max(1,total_tokens),3),
        "chars_per_token":round(total_chars/max(1,total_tokens),3),
        "median_passage_tokens":lengths[len(lengths)//2],
        "p95_passage_tokens":lengths[(95*len(lengths)-1)//100],
        "passages_over_512_with_specials":overflow,
        "roundtrip_passages":len(texts),
        "tokenizer_bytes":target.stat().st_size,
        "train_seconds":round(elapsed,3),
    }
    report["examples"] = []
    for example in ("Скалярное произведение", "перпендикулярность векторов",
                    r"\vec a\cdot\vec b=|\vec a||\vec b|\cos\alpha",
                    "Ёж, λ, ∀x∈ℝ 🧪\n"):
        ids = restored.encode(example,add_special=False)
        report["examples"].append({"text":example,"ids":ids,"tokens":len(ids),
             "pieces":[restored.pieces[i].decode("utf-8",errors="backslashreplace") for i in ids],
             "roundtrip":restored.decode(ids)==example})
    (output/"tokenizer-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=Path,default=Path("ai/generated"))
    p.add_argument("--output",type=Path,default=Path("ai/generated"))
    p.add_argument("--vocab-size",type=int,default=16384)
    p.add_argument("--min-frequency",type=int,default=2)
    args=p.parse_args()
    print(json.dumps(run(args.index,args.output,args.vocab_size,args.min_frequency),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
