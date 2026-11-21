"""Step 5b: rank the products that survived the SQL filters. Owner: M2.

v0: keyword overlap (works with zero models installed).
TODO(M2): BM25 (rank_bm25) + e5 embeddings (ai/embeddings.py), fused with rrf_fuse() below.
"""
from __future__ import annotations

import re

from app.schemas import Product

_TOKEN = re.compile(r"[\wऀ-ॿ]+")
_STOP = {
    "i", "me", "my", "a", "an", "the", "for", "to", "of", "and", "or", "in", "on", "with", "who", "that", "is", "are",
    "show", "need", "want", "looking", "please", "some", "any", "under", "below", "above", "rs", "inr", "k", "only",
    "not", "no", "buy", "get", "find", "good", "best", "like", "likes", "mujhe", "chahiye", "ek", "koi", "se", "kam",
    "ka", "ki", "ke", "hai", "do", "you", "sell", "have",
}


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and not t.isdigit()]


def _doc_text(p: Product) -> str:
    return " ".join([p.name, p.name, p.brand or "", p.category, p.subcategory or "", " ".join(p.tags), p.description])


def keyword_rank(query: str, products: list[Product]) -> list[tuple[Product, float]]:
    q = set(tokens(query))
    if not q:  # no content words (e.g. only "cheaper ones"): keep everything, best rated first
        return sorted(((p, 0.0) for p in products), key=lambda x: -(x[0].rating or 0))
    scored = []
    for p in products:
        doc = tokens(_doc_text(p))
        score = sum(doc.count(t) for t in q)
        if score > 0:
            scored.append((p, float(score)))
    return sorted(scored, key=lambda x: (-x[1], -(x[0].rating or 0)))


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def rank(query: str, products: list[Product], sort: str | None = None) -> list[tuple[Product, float]]:
    ranked = keyword_rank(query, products)
    if sort == "price_asc":
        ranked.sort(key=lambda x: (x[0].price_inr is None, x[0].price_inr or 0))
    elif sort == "price_desc":
        ranked.sort(key=lambda x: -(x[0].price_inr or 0))
    elif sort == "rating_desc":
        ranked.sort(key=lambda x: -(x[0].rating or 0))
    return ranked
