"""Versioned local checkpoints; exact tokenizer identity is mandatory."""
import hashlib
from pathlib import Path

import torch

from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder
from ai.tokenizer.bpe import BPETokenizer

FORMAT_VERSION = 1

def tokenizer_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_checkpoint(path, model, tokenizer_path, *, trained_steps=0):
    if type(trained_steps) is not int or trained_steps < 0:
        raise ValueError("invalid trained_steps")
    if BPETokenizer.load(Path(tokenizer_path)).vocab_size != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    torch.save({"format_version":FORMAT_VERSION,
                "config":model.config.to_dict(),
                "tokenizer_sha256":tokenizer_sha256(tokenizer_path),
                "trained_steps":trained_steps,
                "state_dict":model.state_dict()},tmp)
    tmp.replace(path)

def load_checkpoint(path, tokenizer_path, *, map_location="cpu", require_trained=True):
    data = torch.load(path,map_location=map_location,weights_only=True)
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    if data.get("tokenizer_sha256") != tokenizer_sha256(tokenizer_path):
        raise ValueError("tokenizer hash mismatch")
    steps = data.get("trained_steps")
    if type(steps) is not int or steps < 0 or (require_trained and steps == 0):
        raise ValueError("checkpoint is untrained or invalid")
    config = EncoderConfig(**data["config"])
    if BPETokenizer.load(Path(tokenizer_path)).vocab_size != config.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    model = UniversityEncoder(config)
    model.load_state_dict(data["state_dict"],strict=True)
    return model.to(map_location).eval(), steps
