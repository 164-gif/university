"""Dynamic right padding and tokenizer/model compatibility checks."""
import torch

from ai.tokenizer.bpe import PAD, BOS, EOS

def collate_texts(tokenizer, texts, config, *, device=None, pad_to=None):
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match encoder")
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not one string")
    rows = [tokenizer.encode(text) for text in texts]
    if not rows:
        raise ValueError("empty batch")
    length = max(map(len, rows))
    if length > config.max_length:
        raise ValueError("text exceeds model context; split it before encoding")
    if pad_to is not None:
        if not length <= pad_to <= config.max_length:
            raise ValueError("invalid padding length")
        length = pad_to
    ids = [row + [PAD]*(length-len(row)) for row in rows]
    masks = [[1]*len(row)+[0]*(length-len(row)) for row in rows]
    return {"input_ids":torch.tensor(ids,dtype=torch.long,device=device),
            "attention_mask":torch.tensor(masks,dtype=torch.long,device=device)}
