"""Explicit multi-head bidirectional self-attention (no causal mask)."""
import math

import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask):
        batch, length, hidden = x.shape
        q, k, v = self.qkv(x).reshape(
            batch, length, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)
        logits = (q @ k.transpose(-2, -1)) * self.scale
        # Ignore padded keys. Finite sentinel also keeps all-masked rows
        # finite; such rows are explicitly zeroed below.
        keep = attention_mask[:, None, None, :].bool()
        weights = torch.softmax(logits.masked_fill(~keep, -1e4), dim=-1)
        weights = weights * keep.to(weights.dtype)
        weights = self.dropout(weights)
        out = (weights @ v).transpose(1, 2).reshape(batch, length, hidden)
        return self.output(out) * attention_mask.unsqueeze(-1).to(out.dtype)
