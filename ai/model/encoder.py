"""A compact trainable bidirectional Transformer with masked mean pooling."""
import torch
from torch import nn
from torch.nn import functional as F

from ai.model.attention import SelfAttention
from ai.model.config import EncoderConfig

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attention = SelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask):
        x = x + self.dropout(self.attention(self.norm1(x), attention_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x * attention_mask.unsqueeze(-1).to(x.dtype)

class UniversityEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=config.pad_id)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.projection = nn.Linear(config.hidden_size, config.embedding_size)
        self.apply(self._initialize)
        with torch.no_grad():
            self.token_embedding.weight[config.pad_id].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def hidden_states(self, input_ids, attention_mask):
        """Return [batch, sequence, hidden_size] contextual states.

        Contract: right-padded rank-2 int64 IDs and matching binary mask,
        length <= max_length. Use collate_texts for validated text inputs.
        All-masked rows return zero instead of NaN.
        """
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        mask = attention_mask.bool()
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None,:,:]
        x = self.dropout(self.input_norm(x))
        valid = mask.unsqueeze(-1).to(x.dtype)
        x = x * valid
        for block in self.blocks:
            x = block(x, mask)
        return self.final_norm(x)

    def forward(self, input_ids, attention_mask):
        x = self.hidden_states(input_ids, attention_mask)
        mask = attention_mask.bool()
        valid = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        projected = self.projection(pooled)
        # Normalize in FP32 for numerical stability under mixed precision.
        result = F.normalize(projected.float(), p=2, dim=-1, eps=1e-8)
        return result * mask.any(dim=1, keepdim=True).to(result.dtype)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
