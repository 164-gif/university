"""Build static passage metadata and embeddings for offline browser search."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ai.model.checkpoint import load_checkpoint
from ai.model.inputs import collate_texts
from ai.tokenizer.bpe import BPETokenizer


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    base = args.directory
    tokenizer_path = base / "tokenizer.json"
    model, _ = load_checkpoint(base / "trained/encoder-best.pt", tokenizer_path, map_location=device)
    tokenizer = BPETokenizer.load(tokenizer_path)
    rows = []
    for split in ("train", "validation", "test"):
        rows.extend(json.loads(line) for line in (base / f"dataset/{split}-passages.jsonl").read_text(encoding="utf-8").splitlines() if line)
    vectors = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            batch = collate_texts(tokenizer, [row["encoder_text"] for row in batch_rows], model.config, device=device)
            vectors.append(model(**batch).float().cpu().numpy())
    embeddings = np.concatenate(vectors).astype("<f4", copy=False)
    args.output.mkdir(parents=True, exist_ok=True)
    passages_path = args.output / "passages.json"
    embeddings_path = args.output / "embeddings.f32"
    for path, data in ((passages_path, json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
                       (embeddings_path, embeddings.tobytes(order="C"))):
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(data)
        temp.replace(path)
    manifest = {"format": "university-browser-index-v1", "rows": len(rows), "dimensions": int(embeddings.shape[1]),
                "dtype": "float32-little-endian", "checkpoint_sha256": sha256(base / "trained/encoder-best.pt"),
                "tokenizer_sha256": sha256(tokenizer_path), "passages_sha256": sha256(passages_path),
                "embeddings_sha256": sha256(embeddings_path), "passages_bytes": passages_path.stat().st_size,
                "embeddings_bytes": embeddings_path.stat().st_size}
    manifest_path = args.output / "manifest.json"
    temp = manifest_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("ai/generated/full"))
    parser.add_argument("--output", type=Path, default=Path("ai/generated/full/browser"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    run(parser.parse_args())
