"""Build deterministic JSONL passages without changing any teaching material."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from ai.indexing.chunker import SCHEMA_VERSION, chunk_document, classify, digest

ROOT = Path(__file__).resolve().parents[2]

def build(root: Path, output: Path, target: int = 1000) -> dict:
    root = root.resolve()
    output = output.resolve()
    for name in ("subjects", "cohorts"):
        protected = root / name
        if output == protected or protected in output.parents:
            raise ValueError("output must not be inside teaching materials")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    sources = {}
    for directory in ("subjects", "cohorts"):
        for p in sorted((root / directory).rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            path = p.relative_to(root).as_posix()
            if classify(path) is None:
                continue
            source = p.read_bytes().decode("utf-8")
            chunks = chunk_document(path, source, target)
            if chunks:
                sources[path] = digest(source)
                records.extend(chunks)
    records.sort(key=lambda r: (r["source"], r["line_start"]))
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate passage id")
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "target_chars": target,
        "documents": len(sources),
        "passages": len(records),
        "layers": dict(sorted(Counter(r["layer"] for r in records).items())),
        "cohorts": dict(sorted(Counter(r["cohort"] for r in records).items())),
        "oversized_passages": sum(len(r["text"]) > target for r in records),
        "index_sha256": digest(data),
        "sources": sources,
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = None
    # Replace each generated file atomically. Manifest hash allows consumers
    # to detect an interrupted/mismatched build.
    for name, content in (
        ("passages.jsonl", data),
        ("manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"),
    ):
        tmp = output / (name + ".tmp")
        tmp.write_bytes(content.encode("utf-8"))
        tmp.replace(output / name)
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "ai" / "generated")
    parser.add_argument("--target-chars", type=int, default=1000)
    args = parser.parse_args()
    if args.target_chars < 1:
        parser.error("--target-chars must be positive")
    m = build(args.root, args.output, args.target_chars)
    print(json.dumps({k: v for k, v in m.items() if k != "sources"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
