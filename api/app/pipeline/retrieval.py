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
    # request words that say nothing about the product ("give me options for shoes" must search for "shoes" only)
    "give", "options", "option", "something", "products", "product", "items", "item", "suggest", "recommend",
    "ideas", "suggestions", "dikhao", "dikhaiye", "batao", "can", "could", "would", "what", "which",
    # Hinglish filler ("papa ke liye kuch accha sa perfume dikhao")
    "kuch", "accha", "acha", "achha", "sa", "si", "papa", "mummy", "mere", "meri", "mera", "dena", "chaiye",
    "chahie", "wale", "andar", "sasta", "sasti", "saste", "bhai", "behen", "dost", "yaar", "zara", "thoda",
    "karo", "kariye", "kijiye", "karna", "suggest", "budget", "hona", "chahta", "chahti", "hoon", "hu", "ho",
}


def tokens(text: str) -> list[str]:
    """Tokenize text into lowercased content tokens (English + Devanagari Hindi)."""
    return [t for t in _TOKEN.findall(text.lower())
            if t not in _STOP and not any(ch.isdigit() for ch in t) and not _filler_typo(t)]


_LONG_FILLER = [w for w in ("options", "suggest", "something", "products", "recommend", "please", "looking",
                            "suggestions", "anything", "chahiye", "dikhao", "dikhaiye", "available")]


def _filler_typo(t: str) -> bool:
    """'optons', 'sugest', 'somthing': a misspelt request word, not a product ("show"/"shoe" stay apart: only
    words of 6+ letters are checked)."""
    if len(t) < 5:
        return False
    return any(t[0] == w[0] and abs(len(t) - len(w)) <= 2 and DamerauLevenshtein.distance(t, w) <= (1 if len(w) < 7 else 2)
               for w in _LONG_FILLER)


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
    # typo tolerance is only for words the catalogue does not contain ("nkie"); a real word like "mat" or
    # "power" must not fuzzily match "meat" or "powder"
    known_words = {t for t in q_tokens if indexer.get_bm25_scores([t])}
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
                    if len(qt) < 4 or qt in known_words:
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
                        if len(qt) < 4 or qt in known_words:
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


# e5 gives EVERY product a positive cosine (~0.75 even for unrelated items), so "score > 0" lets the whole
# catalogue through and a price sort then surfaces lemons for "cheaper trekking gear". Real matches sit within
# a few hundredths of the best score over the WHOLE catalogue (measured: 0.84-0.90 best, unrelated ~0.05 lower),
# so we keep only products within DENSE_MARGIN of that global best. Global, not per filtered set: after a
# budget filter the best remaining item may itself be unrelated.
DENSE_MARGIN = float(os.getenv("DENSE_MARGIN", "0.04"))
# In hybrid mode, when BM25 already found keyword matches, meaning-only matches must be closer to the best one:
# "cheaper shoes" sorted by price otherwise surfaces a hand soap that scored 0.806 vs the best 0.837.
HYBRID_DENSE_MARGIN = float(os.getenv("HYBRID_DENSE_MARGIN", "0.025"))
# Measured on our catalogue: products we sell score >= 0.825 at best; things we don't sell can score 0.78-0.835.
DENSE_FLOOR = float(os.getenv("DENSE_FLOOR", "0.80"))
# With an unknown product word in the request, meaning search must be at least this sure (measured: wallet 0.816,
# gaming console 0.825, smartwatch ECG 0.835 are not sold; sold items with odd words: nail paint 0.843).
DENSE_CONFIDENT = float(os.getenv("DENSE_CONFIDENT", "0.84"))
_NOT_PRODUCT_WORDS = {
    "coding", "programming", "gaming", "office", "work", "study", "college", "school", "students", "student",
    "travel", "travelling", "traveling", "trip", "daily", "everyday", "casual", "formal", "party", "wedding",
    "festive", "father", "mother", "dad", "mom", "wife", "husband", "brother", "sister", "friend", "kids", "child",
    "son", "daughter", "boyfriend", "girlfriend", "gift", "gifts", "present", "likes", "loves", "lover", "home",
    "summer", "winter", "monsoon", "rainy", "beginners", "beginner", "professional", "long", "walks", "good",
    "comfortable", "useful", "nice", "great", "durable", "lightweight", "portable", "cheap", "affordable",
}


def dense_rank(query: str, products: list[Product], margin: Optional[float] = None) -> list[tuple[Product, float]]:
    """Dense semantic retrieval using multilingual sentence embeddings (relevance-gated)."""
    if not products or not query.strip():
        return sorted(((p, 0.0) for p in products), key=lambda x: -(x[0].rating or 0))

    # embed only the content words: Hinglish filler ("papa ke liye kuch accha sa") blurs the meaning
    dense_scores = indexer.get_dense_scores(" ".join(tokens(query)) or query)
    if not dense_scores:
        return []
    best = max(dense_scores.values())
    if best < DENSE_FLOOR:
        return []   # nothing in the catalogue is even close ("mosquito net": 0.78; real matches start ~0.825)
    cutoff = best - (DENSE_MARGIN if margin is None else margin)

    scored = []
    for p in products:
        score = dense_scores.get(p.product_id, 0.0)
        if score > 0 and score >= cutoff:
            scored.append((p, float(score)))

    if not scored:
        return []

    return sorted(scored, key=lambda x: (-x[1], -(x[0].rating or 0), x[0].product_id))


def _known_word(t: str) -> bool:
    """Does any catalogue product contain this word (or its singular / plural)?"""
    variants = {t, t[:-1] if t.endswith("s") else t + "s", t[:-2] if t.endswith("es") else t}
    return any(indexer.get_bm25_scores([v]) for v in variants if len(v) > 1)


def probably_not_sold(shopper_text: str) -> bool:
    """The SHOPPER's words (not our added synonyms) name a product word that appears nowhere in the catalogue
    ("wallet", "console", "ecg") and meaning search is not clearly confident -> we most likely do not sell it.
    Use/recipient words ("coding", "father") are not product words."""
    words = tokens(shopper_text)
    unknown = [t for t in words if t not in _NOT_PRODUCT_WORDS and len(t) > 2 and not _known_word(t)]
    if not unknown:
        return False
    dense_all = indexer.get_dense_scores(" ".join(words))
    return bool(dense_all) and max(dense_all.values()) < DENSE_CONFIDENT


def _strong_keyword_match(q_tokens: list[str], bm25_results: list[tuple[Product, float]]) -> bool:
    """Some product contains EVERY query word ("basketball net" would not count for "mosquito net")."""
    if not q_tokens:
        return False
    need = set(q_tokens)
    return any(need <= set(tokens(_doc_text(p))) for p, s in bm25_results if s > 0)


def hybrid_rank(query: str, products: list[Product], k: int = 60) -> list[tuple[Product, float]]:
    """Hybrid retrieval combining BM25 keyword matching and multilingual dense embeddings via RRF."""
    if not products:
        return []

    bm25_results = bm25_rank(query, products)
    bm25_matched = bool(bm25_results) and any(s > 0 for _, s in bm25_results)
    # Do the query words name real products ANYWHERE in the catalogue ("laptop")? Then only strong meaning
    # matches may join, even if the budget filter removed every keyword match: "cheaper laptops" must
    # come back empty, not with a speaker. Otherwise (Hindi wording, "long walks") meaning search leads.
    q_tokens = tokens(query)
    words_known = bm25_matched or bool(q_tokens and indexer.get_bm25_scores(q_tokens))

    dense_results = dense_rank(query, products, margin=HYBRID_DENSE_MARGIN if words_known else None)

    # If dense is unavailable, fallback to BM25 - unless dense ran and found nothing close (below the floor)
    if not dense_results:
        dense_all = indexer.get_dense_scores(" ".join(q_tokens) or query)
        if dense_all and max(dense_all.values()) < DENSE_FLOOR and not _strong_keyword_match(q_tokens, bm25_results):
            return []   # nothing in the catalogue is close in meaning, and no product has all the words
        return bm25_results

    # If BM25 has no query matches, fallback to dense
    if not bm25_matched:
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
