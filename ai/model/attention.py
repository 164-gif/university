"""Explicit multi-head bidirectional self-attention (no causal mask)."""
import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask):
        batch, length, hidden = x.shape
        q, k, v = self.qkv(x).reshape(
            batch, length, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)
        # PyTorch SDPA selects Flash, memory-efficient, or math attention
        # according to device, dtype, shape, and the active backend settings.
        # Boolean True means a key is allowed. Explicitly zero padded query
        # rows, preserving the previous implementation's all-masked behavior.
        keep = attention_mask[:, None, None, :].bool()
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=keep,
            dropout_p=self.dropout.p if self.training else 0.0,
        ).transpose(1, 2).reshape(batch, length, hidden)
        return self.output(out) * attention_mask.unsqueeze(-1).to(out.dtype)
