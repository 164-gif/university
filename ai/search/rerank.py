"""Rescore a fixed list of BM25F hits with a compact encoder."""


def rerank(hits, similarities, weight):
    if not hits:
        return []
    scale = hits[0].score
    scored = [(hit, hit.score / scale + weight * similarities[hit.passage["id"]])
              for hit in hits]
    return sorted(scored, key=lambda item: -item[1])
