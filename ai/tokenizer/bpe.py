"""Deterministic byte-level BPE for the University encoder; stdlib only."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import heapq
import json
from pathlib import Path
import re

PAD, BOS, EOS, MASK = range(4)
SPECIAL = ("[PAD]", "[BOS]", "[EOS]", "[MASK]")
BYTE_OFFSET = len(SPECIAL)
BASE_SIZE = BYTE_OFFSET + 256
VERSION = 1
# Preserve every byte; never lowercase, strip whitespace or normalize Unicode.
# TeX commands are separate segments; math punctuation remains recoverable.
SEGMENTS = re.compile(r"\\[A-Za-z]+|[^\W\d_]+|\d+|[ \t]+|\r\n|[^\w\s]+|[\s\S]", re.UNICODE)

def split(text: str):
    return SEGMENTS.findall(text)

def byte_ids(text: str) -> tuple[int, ...]:
    return tuple(b + BYTE_OFFSET for b in text.encode("utf-8"))

def replace_pair(seq, pair, new):
    out = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i+1]) == pair:
            out.append(new)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)

class BPETokenizer:
    def __init__(self, merges=()):
        self.merges = tuple(tuple(p) for p in merges)
        self.ranks = {}
        self.pieces = [None]*BYTE_OFFSET + [bytes([i]) for i in range(256)]
        for i, pair in enumerate(self.merges, BASE_SIZE):
            if (len(pair) != 2 or any(type(x) is not int or x < BYTE_OFFSET or x >= i for x in pair)
                    or pair in self.ranks):
                raise ValueError("invalid BPE merge")
            self.ranks[pair] = i
            self.pieces.append(self.pieces[pair[0]] + self.pieces[pair[1]])

    @property
    def vocab_size(self):
        return len(self.pieces)

    @lru_cache(maxsize=8192)
    def encode_segment(self, segment):
        seq = byte_ids(segment)
        while len(seq) > 1:
            best = min((self.ranks.get(p, self.vocab_size) for p in zip(seq, seq[1:])),
                       default=self.vocab_size)
            if best == self.vocab_size:
                break
            seq = replace_pair(seq, self.merges[best-BASE_SIZE], best)
        return seq

    def encode(self, text, *, add_special=True):
        ids = [i for segment in split(text) for i in self.encode_segment(segment)]
        return [BOS, *ids, EOS] if add_special else ids

    def decode(self, ids, *, errors="strict"):
        parts = []
        for i in ids:
            if type(i) is not int or i < 0 or i >= self.vocab_size:
                raise ValueError("invalid token id")
            if i >= BYTE_OFFSET:
                parts.append(self.pieces[i])
        return b"".join(parts).decode("utf-8", errors=errors)

    def model_inputs(self, text, *, max_length=512, truncate=False):
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        ids = self.encode(text, add_special=False)
        overflow = len(ids) > max_length-2
        if overflow and not truncate:
            raise ValueError("text exceeds context; split it or opt into truncation")
        ids = [BOS, *ids[:max_length-2], EOS]
        padding = max_length-len(ids)
        return {"input_ids":ids+[PAD]*padding,
                "attention_mask":[1]*len(ids)+[0]*padding,
                "truncated":overflow}

    def save(self, path: Path):
        data = {"version":VERSION, "algorithm":"byte-bpe", "special":list(SPECIAL),
                "vocab_size":self.vocab_size, "merges":self.merges}
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",",":"))+"\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("version") != VERSION or data.get("algorithm") != "byte-bpe"
                or data.get("special") != list(SPECIAL)):
            raise ValueError("unsupported tokenizer format")
        obj = cls(data["merges"])
        if obj.vocab_size != data["vocab_size"]:
            raise ValueError("tokenizer size mismatch")
        return obj

def train(texts, *, vocab_size=16384, min_frequency=2):
    """Weighted pair counts, inverted word lists and a lazy max heap.
    Tie breaking is lexicographic by pair ID; no random seed is needed.
    """
    if not BASE_SIZE <= vocab_size <= 65536 or min_frequency < 1:
        raise ValueError("invalid vocabulary size or minimum frequency")
    frequencies = Counter(segment for text in texts for segment in split(text))
    words = [byte_ids(s) for s in sorted(frequencies)]
    weights = [frequencies[s] for s in sorted(frequencies)]
    pair_counts = Counter()
    locations = defaultdict(set)
    for i, word in enumerate(words):
        for p, n in Counter(zip(word, word[1:])).items():
            pair_counts[p] += n*weights[i]
            locations[p].add(i)
    heap = [(-n,p) for p,n in pair_counts.items() if n >= min_frequency]
    heapq.heapify(heap)
    merges = []
    while heap and BASE_SIZE+len(merges) < vocab_size:
        neg, pair = heapq.heappop(heap)
        if -neg != pair_counts[pair] or -neg < min_frequency:
            continue
        new = BASE_SIZE+len(merges)
        touched = set()
        for i in sorted(locations[pair]):
            old = words[i]
            updated = replace_pair(old,pair,new)
            if old == updated:
                continue
            before = Counter(zip(old,old[1:]))
            after = Counter(zip(updated,updated[1:]))
            for p in before.keys() | after.keys():
                delta = after[p]-before[p]
                if delta:
                    pair_counts[p] += delta*weights[i]
                    touched.add(p)
                if after[p]:
                    locations[p].add(i)
                else:
                    locations[p].discard(i)
            words[i] = updated
        merges.append(pair)
        for p in sorted(touched):
            if pair_counts[p] >= min_frequency:
                heapq.heappush(heap,(-pair_counts[p],p))
    return BPETokenizer(merges)
