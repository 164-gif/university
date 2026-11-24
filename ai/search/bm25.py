"""BM25F lexical retrieval; results are candidates, not verified answers."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

from ai.indexing.chunker import DEFAULT_COHORT, SCHEMA_VERSION, digest
from ai.search.lexical import tokens, near

FIELDS = {"body": (1.0, .75), "heading": (1.65, .25), "title": (.65, .25)}
DEFAULT_LAYERS = frozenset({"notes", "concepts", "exam", "tests", "laboratory_physics",
                            "teacher_questions", "quick_cr_prep", "practice"})
K1 = 1.2

@dataclass(frozen=True)
class Hit:
    passage: dict
    score: float
    coverage: float
    matched: tuple[str, ...]
    corrected: tuple[tuple[str,str], ...]

class BM25:
    def __init__(self, records: list[dict]):
        self.records = tuple(records)
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("duplicate passage id")
        self.postings = defaultdict(dict)
        self.lengths = []
        self.average = {}
        totals = Counter()
        for i, row in enumerate(records):
            fields = {
                "body": tokens(row["text"]),
                "heading": tokens(row["headings"][-1] if row["headings"] else ""),
                "title": tokens(row["title"]),
            }
            lengths = {f: len(t) for f,t in fields.items()}
            self.lengths.append(lengths)
            totals.update(lengths)
            for f, terms in fields.items():
                for term, count in Counter(terms).items():
                    self.postings[term].setdefault(i, {})[f] = count
        self.average = {f: max(1, totals[f]/max(1,len(records))) for f in FIELDS}
        self.by_length = defaultdict(list)
        for term in sorted(self.postings):
            if len(term) >= 4 and term.isalpha():
                self.by_length[len(term)].append(term)

    @classmethod
    def load(cls, directory: Path) -> BM25:
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        raw = (directory/"passages.jsonl").read_bytes().decode("utf-8")
        if manifest["schema_version"] != SCHEMA_VERSION or digest(raw) != manifest["index_sha256"]:
            raise ValueError("index schema or integrity mismatch; rebuild the index")
        records = [json.loads(line) for line in raw.splitlines() if line]
        if len(records) != manifest["passages"]:
            raise ValueError("index count mismatch")
        return cls(records)

    def search(self, question: str, *, cohort: str | None = DEFAULT_COHORT,
               subject: str | None = None, layers: set[str] | frozenset[str] | None = None,
               limit: int = 5, fuzzy: bool = True) -> list[Hit]:
        if limit < 1 or not question.strip() or len(question) > 4096:
            return []
        layers = DEFAULT_LAYERS if layers is None else layers
        query = tuple(dict.fromkeys(tokens(question)))
        if not query:
            return []
        eligible = {i for i,r in enumerate(self.records)
                    if (cohort is None or r["cohort"] == cohort)
                    and (subject is None or r["subject"] == subject)
                    and r["layer"] in layers}
        n = len(eligible)
        if not n:
            return []
        scores = Counter()
        matched = defaultdict(set)
        corrections = defaultdict(set)
        for term in query:
            exact = self.postings.get(term, {})
            options = [(term, 1.0)] if eligible.intersection(exact) else []
            if not options and fuzzy and len(term) >= 5 and term.isalpha():
                candidates = []
                for length in (len(term)-1,len(term),len(term)+1):
                    for alt in self.by_length[length]:
                        if near(term, alt) and eligible.intersection(self.postings[alt]):
                            candidates.append(alt)
                # Ambiguous corrections are not guessed.
                if len(candidates) == 1:
                    options = [(candidates[0], .65)]
            for alt, factor in options:
                postings = self.postings[alt]
                ids = eligible.intersection(postings)
                idf = math.log1p((n-len(ids)+.5)/(len(ids)+.5))
                for i in ids:
                    tf = 0.0
                    for field,count in postings[i].items():
                        w,b = FIELDS[field]
                        norm = 1-b+b*self.lengths[i][field]/self.average[field]
                        tf += w*count/norm
                    scores[i] += factor*idf*(K1+1)*tf/(K1+tf)
                    matched[i].add(term)
                    if alt != term:
                        corrections[i].add((term,alt))
        ranked = []
        for i,score in scores.items():
            coverage = len(matched[i])/len(query)
            # Require at least half of the meaningful query terms; do not
            # interpret this heuristic or the BM25 score as a probability.
            if coverage < .5:
                continue
            ranked.append(Hit(self.records[i],score*coverage,coverage,
                              tuple(sorted(matched[i])),tuple(sorted(corrections[i]))))
        ranked.sort(key=lambda h: (-h.score,h.passage["source"],h.passage["line_start"]))
        # Suppress byte-identical passages in the same subject/cohort/layer,
        # never collapse different source layers or courses.
        seen = set()
        out = []
        for hit in ranked:
            r = hit.passage
            key = (r["cohort"],r["subject"],r["layer"],r["text_sha256"])
            if key in seen:
                continue
            seen.add(key)
            out.append(hit)
            if len(out) == limit:
                break
        return out
