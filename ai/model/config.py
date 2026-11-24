"""Architecture contract for University Encoder v0.1."""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class EncoderConfig:
    vocab_size: int
    max_length: int = 512
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 512
    embedding_size: int = 128
    dropout: float = 0.1
    pad_id: int = 0

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (
            self.vocab_size, self.max_length, self.hidden_size, self.num_layers,
            self.num_heads, self.intermediate_size, self.embedding_size
        )):
            raise ValueError("dimensions must be positive integers")
        if self.max_length < 2 or self.hidden_size % self.num_heads:
            raise ValueError("invalid context or attention head dimensions")
        if self.pad_id != 0 or self.vocab_size < 260:
            raise ValueError("expected University byte-BPE vocabulary")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self):
        return asdict(self)
