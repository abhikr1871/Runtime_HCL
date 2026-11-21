"""In-memory search index, refreshed after /ingest. Owner: M2.

Maintains:
  - BM25 index over name + brand + category + subcategory + tags + description (rank_bm25)
  - e5 embeddings stored in the `embeddings` SQLite table, cached in an in-memory NumPy matrix
  - refresh(product_ids) re-embeds only changed rows and swaps indexes atomically
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.ai import embeddings
from app.config import get_settings
from app.schemas import Product
from app.store import db, repository

log = logging.getLogger("indexer")

_TOKEN = re.compile(r"[\wऀ-ॿ]+")
_STOP = {
    "i", "me", "my", "a", "an", "the", "for", "to", "of", "and", "or", "in", "on", "with", "who", "that", "is", "are",
    "show", "need", "want", "looking", "please", "some", "any", "under", "below", "above", "rs", "inr", "k", "only",
    "not", "no", "buy", "get", "find", "good", "best", "like", "likes", "mujhe", "chahiye", "ek", "koi", "se", "kam",
    "ka", "ki", "ke", "hai", "do", "you", "sell", "have", "tak", "mein", "aur", "bhi", "liye", "wala", "wali",
}

_lock = threading.Lock()
_products: dict[str, Product] = {}
_bm25_ids: list[str] = []
_bm25_index: Optional[BM25Okapi] = None
_embedding_ids: list[str] = []
_embedding_matrix: Optional[np.ndarray] = None


def tokenize(text: str) -> list[str]:
    """Tokenize query and documents with English + Devanagari support, stopwords removed."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and not t.isdigit()]


def doc_text(p: Product) -> str:
    """Text representation for BM25 and multilingual dense embedding."""
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


def _build_bm25(prods: dict[str, Product]) -> tuple[list[str], Optional[BM25Okapi]]:
    pids = list(prods.keys())
    if not pids:
        return [], None
    corpus = [tokenize(doc_text(prods[pid])) for pid in pids]
    return pids, BM25Okapi(corpus)


def _load_or_compute_embeddings(prods: dict[str, Product]) -> tuple[list[str], Optional[np.ndarray]]:
    s = get_settings()
    if not s.enable_embeddings:
        return [], None

    try:
        rows = db.query("SELECT product_id, vector FROM embeddings WHERE kind='text' AND model=?", (s.embed_model,))
        existing: dict[str, np.ndarray] = {
            r["product_id"]: np.frombuffer(r["vector"], dtype=np.float32) for r in rows
        }

        missing_pids = [pid for pid in prods if pid not in existing]
        if missing_pids:
            log.info("Computing embeddings for %d products with model %s", len(missing_pids), s.embed_model)
            texts = [doc_text(prods[pid]) for pid in missing_pids]
            vecs = embeddings.embed_passages(texts)
            if vecs is not None:
                for pid, vec in zip(missing_pids, vecs):
                    vec_f32 = np.array(vec, dtype=np.float32)
                    existing[pid] = vec_f32
                    db.execute(
                        "INSERT OR REPLACE INTO embeddings (product_id, kind, model, vector, updated_at) "
                        "VALUES (?, 'text', ?, ?, datetime('now'))",
                        (pid, s.embed_model, vec_f32.tobytes()),
                    )

        emb_ids = [pid for pid in prods if pid in existing]
        if not emb_ids:
            return [], None
        matrix = np.array([existing[pid] for pid in emb_ids], dtype=np.float32)
        return emb_ids, matrix
    except Exception as exc:
        log.warning("Dense embeddings indexing skipped or failed: %s", exc)
        return [], None


def rebuild() -> int:
    """Full index rebuild on startup or test reset."""
    global _products, _bm25_ids, _bm25_index, _embedding_ids, _embedding_matrix
    fresh = {p.product_id: p for p in repository.find_products()}
    bm25_ids, bm25_idx = _build_bm25(fresh)
    emb_ids, emb_matrix = _load_or_compute_embeddings(fresh)

    with _lock:
        _products = fresh
        _bm25_ids = bm25_ids
        _bm25_index = bm25_idx
        _embedding_ids = emb_ids
        _embedding_matrix = emb_matrix

    log.info("Indexed %d products (BM25: %d, Dense: %d)", len(fresh), len(bm25_ids), len(emb_ids))
    return len(fresh)


def refresh(product_ids: list[str]) -> None:
    """Called by /ingest so new or changed products are searchable immediately on next request."""
    global _products, _bm25_ids, _bm25_index, _embedding_ids, _embedding_matrix
    updated_prods = repository.get_products(product_ids)
    if not updated_prods:
        return

    with _lock:
        prods_copy = dict(_products)
        for p in updated_prods:
            prods_copy[p.product_id] = p
        _products = prods_copy

    bm25_ids, bm25_idx = _build_bm25(prods_copy)
    emb_ids, emb_matrix = _load_or_compute_embeddings(prods_copy)

    with _lock:
        _bm25_ids = bm25_ids
        _bm25_index = bm25_idx
        _embedding_ids = emb_ids
        _embedding_matrix = emb_matrix


def get_bm25_scores(query_tokens: list[str]) -> dict[str, float]:
    """Scores all indexed products against query tokens using BM25Okapi."""
    with _lock:
        if _bm25_index is None or not query_tokens or not _bm25_ids:
            return {}
        scores = _bm25_index.get_scores(query_tokens)
        return {pid: float(s) for pid, s in zip(_bm25_ids, scores) if s > 0}


def get_dense_scores(query_text: str) -> dict[str, float]:
    """Cosine similarity scores against multilingual e5 embeddings."""
    with _lock:
        if _embedding_matrix is None or len(_embedding_ids) == 0:
            return {}
        q_vec = embeddings.embed_query(query_text)
        if q_vec is None:
            return {}
        q_vec = np.array(q_vec, dtype=np.float32)
        sims = np.dot(_embedding_matrix, q_vec)
        return {pid: float(s) for pid, s in zip(_embedding_ids, sims) if s > 0}


def get_product(product_id: str) -> Optional[Product]:
    with _lock:
        return _products.get(product_id)


def status() -> str:
    with _lock:
        n = len(_products)
        has_emb = _embedding_matrix is not None and len(_embedding_ids) > 0
        if has_emb:
            return f"hybrid (BM25 + Dense {len(_embedding_ids)}/{n} prods)"
        return f"bm25 ({n} products)"
