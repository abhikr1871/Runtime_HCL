"""Step 5b: rank the products that survived the SQL filters. Owner: M2.

Provides:
  - BM25 keyword retrieval (rank_bm25 with typo tolerance)
  - Multilingual dense retrieval (multilingual-e5-small)
  - Hybrid retrieval using Reciprocal Rank Fusion (RRF)
  - Configurable retrieval mode: RETRIEVAL_MODE=bm25|dense|hybrid
  - Unified search_products(filters, query, top_k) interface
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from rapidfuzz.distance import DamerauLevenshtein

from app.config import get_settings
from app.pipeline import filters as filters_mod
from app.schemas import Product
from app.store import indexer

log = logging.getLogger("retrieval")

_TOKEN = re.compile(r"[\wऀ-ॿ]+")
_STOP = {
    "i", "me", "my", "a", "an", "the", "for", "to", "of", "and", "or", "in", "on", "with", "who", "that", "is", "are",
    "show", "need", "want", "looking", "please", "some", "any", "under", "below", "above", "rs", "inr", "k", "only",
    "not", "no", "buy", "get", "find", "good", "best", "like", "likes", "mujhe", "chahiye", "ek", "koi", "se", "kam",
    "ka", "ki", "ke", "hai", "do", "you", "sell", "have", "tak", "mein", "aur", "bhi", "liye", "wala", "wali",
}


def tokens(text: str) -> list[str]:
    """Tokenize text into lowercased content tokens (English + Devanagari Hindi)."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and not t.isdigit()]


def _doc_text(p: Product) -> str:
    """Combines product fields into weighted searchable text."""
    parts = [p.name, p.name]
    if p.brand:
        parts.append(p.brand)
    parts.append(p.category)
    if p.subcategory:
        parts.append(p.subcategory)
    if p.tags:
        parts.extend(p.tags)
    if p.description:
        parts.append(p.description)
    return " ".join(parts)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def bm25_rank(query: str, products: list[Product]) -> list[tuple[Product, float]]:
    """BM25-based keyword ranking over product metadata with typo tolerance."""
    if not products:
        return []
    q_tokens = tokens(query)
    if not q_tokens:
        # No query content tokens (e.g. 'show cheaper ones'): order by rating descending
        return sorted(((p, 0.0) for p in products), key=lambda x: -(x[0].rating or 0))

    indexed_scores = indexer.get_bm25_scores(q_tokens)
    scored: list[tuple[Product, float]] = []

    # If indexer has scores for some candidates, use them; otherwise compute locally
    if indexed_scores and any(p.product_id in indexed_scores for p in products):
        for p in products:
            s = indexed_scores.get(p.product_id, 0.0)
            if s > 0:
                scored.append((p, float(s)))
            else:
                # Typo check: Damerau-Levenshtein distance (handles character swaps like 'nkie' -> 'nike')
                doc_toks = tokens(_doc_text(p))
                fuzzy_bonus = 0.0
                for qt in q_tokens:
                    if len(qt) < 3:
                        continue
                    for dt in doc_toks:
                        if len(dt) >= 3 and qt[0] == dt[0] and DamerauLevenshtein.distance(qt, dt) <= (1 if len(qt) <= 6 else 2):
                            fuzzy_bonus += 2.0
                if fuzzy_bonus > 0:
                    scored.append((p, fuzzy_bonus))
    else:
        # Ad-hoc local BM25 ranking over candidates (e.g. in standalone tests)
        corpus = [tokens(_doc_text(p)) for p in products]
        if any(corpus):
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(q_tokens)
            for p, s in zip(products, scores):
                if s > 0:
                    scored.append((p, float(s)))
                else:
                    # Typo check: Damerau-Levenshtein distance
                    doc_toks = tokens(_doc_text(p))
                    fuzzy_bonus = 0.0
                    for qt in q_tokens:
                        if len(qt) < 3:
                            continue
                        for dt in doc_toks:
                            if len(dt) >= 3 and qt[0] == dt[0] and DamerauLevenshtein.distance(qt, dt) <= (1 if len(qt) <= 6 else 2):
                                fuzzy_bonus += 2.0
                    if fuzzy_bonus > 0:
                        scored.append((p, fuzzy_bonus))

    # If query had content words but nothing matched, return empty so not_found is triggered
    if not scored:
        return []

    return sorted(scored, key=lambda x: (-x[1], -(x[0].rating or 0), x[0].product_id))


def keyword_rank(query: str, products: list[Product]) -> list[tuple[Product, float]]:
    """Backward compatibility alias for bm25_rank."""
    return bm25_rank(query, products)


def dense_rank(query: str, products: list[Product]) -> list[tuple[Product, float]]:
    """Dense semantic retrieval using multilingual sentence embeddings."""
    if not products or not query.strip():
        return sorted(((p, 0.0) for p in products), key=lambda x: -(x[0].rating or 0))

    dense_scores = indexer.get_dense_scores(query)
    if not dense_scores:
        return []

    scored = []
    for p in products:
        score = dense_scores.get(p.product_id, 0.0)
        if score > 0:
            scored.append((p, float(score)))

    if not scored:
        return []

    return sorted(scored, key=lambda x: (-x[1], -(x[0].rating or 0), x[0].product_id))


def hybrid_rank(query: str, products: list[Product], k: int = 60) -> list[tuple[Product, float]]:
    """Hybrid retrieval combining BM25 keyword matching and multilingual dense embeddings via RRF."""
    if not products:
        return []

    bm25_results = bm25_rank(query, products)
    dense_results = dense_rank(query, products)

    # If dense is unavailable, fallback to BM25
    if not dense_results:
        return bm25_results

    # If BM25 has no query matches, fallback to dense
    if not bm25_results or all(s == 0 for _, s in bm25_results):
        return dense_results

    bm25_ids = [p.product_id for p, s in bm25_results if s > 0]
    dense_ids = [p.product_id for p, s in dense_results if s > 0]

    fused_dict = dict(rrf_fuse([bm25_ids, dense_ids], k=k))

    scored: list[tuple[Product, float]] = []
    for p in products:
        score = fused_dict.get(p.product_id, 0.0)
        if score > 0:
            scored.append((p, float(score)))

    if not scored:
        return bm25_results

    return sorted(scored, key=lambda x: (-x[1], -(x[0].rating or 0), x[0].product_id))


def rank(
    query: str,
    products: list[Product],
    sort: str | None = None,
    mode: str | None = None,
) -> list[tuple[Product, float]]:
    """Rank candidate products using the configured retrieval mode (bm25 | dense | hybrid)."""
    if not products:
        return []

    selected_mode = (
        mode
        or os.getenv("RETRIEVAL_MODE")
        or get_settings().retrieval_mode
        or "hybrid"
    ).strip().lower()

    if selected_mode == "bm25":
        ranked = bm25_rank(query, products)
    elif selected_mode == "dense":
        ranked = dense_rank(query, products)
        if not ranked:  # fallback to BM25 if embeddings disabled/missing
            ranked = bm25_rank(query, products)
    else:  # hybrid (default)
        ranked = hybrid_rank(query, products)

    if sort == "price_asc":
        ranked.sort(key=lambda x: (x[0].price_inr is None, x[0].price_inr or 0))
    elif sort == "price_desc":
        ranked.sort(key=lambda x: -(x[0].price_inr or 0))
    elif sort == "rating_desc":
        ranked.sort(key=lambda x: -(x[0].rating or 0))

    return ranked


def search_products(
    filters: dict[str, Any] | None = None,
    query: str = "",
    top_k: int = 10,
    sort: str | None = None,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    """Unified search and retrieval interface contract for M1 and other modules.

    Applies hard SQL filters first, ranks surviving candidates with BM25 / dense / hybrid,
    and returns formatted product records.
    """
    f = filters or {}
    candidates = filters_mod.apply(f)
    if not candidates:
        return []

    actual_mode = (
        mode
        or os.getenv("RETRIEVAL_MODE")
        or get_settings().retrieval_mode
        or "hybrid"
    ).strip().lower()

    ranked = rank(query, candidates, sort=sort or f.get("sort"), mode=actual_mode)
    top_items = ranked[:top_k]

    return [
        {
            "product_id": p.product_id,
            "name": p.name,
            "brand": p.brand,
            "price": p.price_inr,
            "in_stock": p.stock > 0,
            "score": round(score, 4),
            "retrieval_method": actual_mode,
        }
        for p, score in top_items
    ]
