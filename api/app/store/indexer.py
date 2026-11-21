"""Search index and catalogue storage (BM25 + multilingual embeddings). Owner: M2-B.

Coordinates with M2-A by providing:
- get_searchable_products()
- get_product_embeddings()
- get_bm25_documents()
- rebuild_product_index()
- update_product_index(product_id)
- status()
- bm25_search(query, candidate_ids)
- dense_search(query, candidate_ids)

Rules:
- Unchanged products are never unnecessarily re-embedded (tracked by content hash).
- Product ID mapping is strictly preserved across SQLite, BM25, and embeddings.
- Inactive/deleted products are removed from search indexes.
- Thread-safe atomic index swapping.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Any, Optional, Union

import numpy as np
from rank_bm25 import BM25Okapi

from app.ai import embeddings
from app.config import get_settings
from app.schemas import Product
from app.store import repository

log = logging.getLogger("indexer")
_lock = threading.RLock()

# ---------- State ----------
_products: dict[str, Product] = {}
_content_hashes: dict[str, str] = {}

# BM25 state
_bm25: Optional[BM25Okapi] = None
_bm25_ids: list[str] = []
_tokenized_corpus: list[list[str]] = []

# Dense embedding state (in-memory copy for fast cosine search; the persistent copy is in the vector store)
_vectors: Optional[np.ndarray] = None          # (N, D) float32
_vector_ids: list[str] = []
_vector_map: dict[str, np.ndarray] = {}


# ---------- Vector store: where embeddings are persisted (Chroma support: Jeevan, M2-A) ----------
# VECTOR_STORE=sqlite (default): vectors in the SQLite `embeddings` table.
# VECTOR_STORE=chromadb: vectors in a persistent Chroma collection (CHROMA_DIR, default data/chroma),
#   and dense search queries Chroma. SQLite is then NOT written: vectors live in exactly one place.
# If chromadb is not installed or fails to open, it falls back to SQLite and says so in /health.
_chroma_col = None
_chroma_failed = False


def _chroma():
    """The Chroma 'products' collection, opened once and reused (None if not configured or unavailable)."""
    global _chroma_col, _chroma_failed
    s = get_settings()
    if s.vector_store != "chromadb" or _chroma_failed:
        return None
    if _chroma_col is None:
        try:
            import chromadb
            s.chroma_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(s.chroma_dir))
            _chroma_col = client.get_or_create_collection(name="products", metadata={"hnsw:space": "cosine"})
        except Exception as exc:
            _chroma_failed = True
            log.warning("ChromaDB unavailable (%s); storing vectors in SQLite instead", exc)
            return None
    return _chroma_col


def get_chroma_collection():
    return _chroma()


def reset_vector_store() -> None:
    """Forget the open collection (tests and settings changes)."""
    global _chroma_col, _chroma_failed
    _chroma_col, _chroma_failed = None, False


def vector_store_name() -> str:
    return "chromadb" if _chroma() is not None else "sqlite"


def _load_vectors() -> dict[str, np.ndarray]:
    col = _chroma()
    if col is not None:
        got = col.get(include=["embeddings"])
        return {pid: np.asarray(v, dtype=np.float32) for pid, v in zip(got["ids"], got["embeddings"])}
    out: dict[str, np.ndarray] = {}
    for pid, blob in repository.get_all_embeddings("text").items():
        try:
            out[pid] = embeddings.bytes_to_vector(blob)
        except Exception as exc:
            log.warning("error deserializing vector for %s: %s", pid, exc)
    return out


def _save_vectors(items: list[tuple[str, np.ndarray]], products: dict[str, Product]) -> None:
    if not items:
        return
    col = _chroma()
    if col is not None:
        col.upsert(ids=[pid for pid, _ in items], embeddings=[np.asarray(v).tolist() for _, v in items],
                   metadatas=[{"product_id": pid, "category": products[pid].category if pid in products else ""}
                              for pid, _ in items])
        return
    s = get_settings()
    repository.save_embeddings_batch([(pid, "text", s.embed_model, embeddings.vector_to_bytes(v)) for pid, v in items])

# ---------- Tokenization ----------
_TOKEN = re.compile(r"[\wऀ-ॿ]+")
_STOP = {
    # English
    "i", "me", "my", "a", "an", "the", "for", "to", "of", "and", "or", "in", "on", "with", "who", "that", "is", "are",
    "show", "need", "want", "looking", "please", "some", "any", "under", "below", "above", "rs", "inr", "k", "only",
    "not", "no", "buy", "get", "find", "good", "best", "like", "likes", "do", "you", "sell", "have",
    # Hindi / Hinglish
    "mujhe", "chahiye", "ek", "koi", "se", "kam", "ka", "ki", "ke", "hai", "hain", "ko", "par", "me", "mein",
    "aur", "ya", "yeh", "woh", "bhi", "accha", "bahut", "ke", "liye", "tha", "thi", "the",
    "मुझे", "चाहिए", "एक", "कोई", "से", "कम", "का", "की", "के", "है", "हैं", "को", "पर", "में",
    "और", "या", "यह", "वह", "भी", "अच्छा", "बहुत", "लिए", "था", "थी", "थे",
}


def tokenize(text: str) -> list[str]:
    """Extracts non-stop, non-pure-digit tokens supporting English and Hindi (Devanagari)."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and not t.isdigit()]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_bm25_under_lock(active_products: list[Product]) -> None:
    global _bm25, _bm25_ids, _tokenized_corpus
    _bm25_ids = [p.product_id for p in active_products]
    _tokenized_corpus = [tokenize(embeddings.prepare_product_text(p)) for p in active_products]
    if _tokenized_corpus:
        _bm25 = BM25Okapi(_tokenized_corpus)
    else:
        _bm25 = None


def _sync_vectors_under_lock() -> None:
    global _vectors, _vector_ids
    _vector_ids = [pid for pid in _products.keys() if pid in _vector_map]
    if _vector_ids:
        _vectors = np.stack([_vector_map[pid] for pid in _vector_ids], axis=0)
    else:
        _vectors = None


# ---------- Public Interface Contract with M2-A ----------
def get_searchable_products() -> dict[str, Product]:
    """Returns a snapshot of all currently searchable active products."""
    with _lock:
        return dict(_products)


def get_product_embeddings() -> tuple[list[str], Optional[np.ndarray]]:
    """Returns (product_ids, matrix_of_vectors) for active products with embeddings."""
    with _lock:
        return list(_vector_ids), None if _vectors is None else _vectors.copy()


def get_bm25_documents() -> tuple[list[str], list[list[str]]]:
    """Returns (product_ids, tokenized_documents) for BM25 search."""
    with _lock:
        return list(_bm25_ids), list(_tokenized_corpus)


def rebuild_product_index() -> int:
    """Full rebuild of in-memory product catalogue, BM25 index, and embedding vectors.
    Returns number of active products indexed.
    """
    global _products, _content_hashes, _vector_map
    active = repository.find_products("is_active = 1")
    fresh_products = {p.product_id: p for p in active}

    # 1. Load stored embeddings from the vector store (SQLite or Chroma)
    stored = _load_vectors()
    fresh_vector_map: dict[str, np.ndarray] = {}
    fresh_hashes: dict[str, str] = {}

    for pid, p in fresh_products.items():
        doc_text = embeddings.prepare_product_text(p)
        fresh_hashes[pid] = _content_hash(doc_text)
        if pid in stored:
            fresh_vector_map[pid] = stored[pid]

    # 2. Re-embed missing products if embeddings are enabled, and persist them in the vector store
    s = get_settings()
    if s.enable_embeddings:
        missing_pids = [pid for pid in fresh_products.keys() if pid not in fresh_vector_map]
        if missing_pids:
            log.info("generating embeddings for %d missing products", len(missing_pids))
            texts = [embeddings.prepare_product_text(fresh_products[pid]) for pid in missing_pids]
            vecs = embeddings.embed_passages(texts)
            if vecs is not None and len(vecs) == len(missing_pids):
                for pid, vec in zip(missing_pids, vecs):
                    fresh_vector_map[pid] = vec
                _save_vectors(list(zip(missing_pids, vecs)), fresh_products)

    # 3. Atomically swap in the new state under lock
    with _lock:
        _products = fresh_products
        _content_hashes = fresh_hashes
        _vector_map = fresh_vector_map
        _build_bm25_under_lock(active)
        _sync_vectors_under_lock()

    log.info("index rebuilt: %d products, %d bm25 docs, %d embedding vectors",
             len(_products), len(_bm25_ids), len(_vector_ids))
    return len(fresh_products)


# Backward-compatible alias
rebuild = rebuild_product_index


def update_product_index(product_id: Union[str, list[str]]) -> None:
    """Incrementally updates the index for one or more product IDs.
    - If product was deactivated or deleted: removes it from search.
    - If product was modified: updates in-memory copy, recomputes embedding ONLY if text changed.
    - If product was added: adds to BM25 and generates embedding.
    """
    pids = [product_id] if isinstance(product_id, str) else list(product_id)
    if not pids:
        return

    s = get_settings()
    with _lock:
        for pid in pids:
            p = repository.get_product(pid)
            if p is None or not p.is_active:
                # Removed or deactivated
                _products.pop(pid, None)
                _content_hashes.pop(pid, None)
                _vector_map.pop(pid, None)
                continue

            _products[pid] = p
            doc_text = embeddings.prepare_product_text(p)
            new_hash = _content_hash(doc_text)
            old_hash = _content_hashes.get(pid)

            _content_hashes[pid] = new_hash

            # Check if embedding needs recomputation
            need_embed = (pid not in _vector_map) or (old_hash != new_hash)
            if need_embed and s.enable_embeddings:
                vec = embeddings.embed_passages([doc_text])
                if vec is not None and len(vec) == 1:
                    v = vec[0]
                    _vector_map[pid] = v
                    _save_vectors([(pid, v)], _products)

        # Re-sync BM25 and vector matrix
        active_list = list(_products.values())
        _build_bm25_under_lock(active_list)
        _sync_vectors_under_lock()


# Backward-compatible alias
refresh = update_product_index


def status() -> str:
    """Human-readable status for /health and diagnostics."""
    with _lock:
        n_prod = len(_products)
        n_bm25 = len(_bm25_ids)
        n_vec = len(_vector_ids)
        embed_mode = "e5" if n_vec > 0 else "disabled"
    return f"bm25+{embed_mode} ({n_prod} products, {n_bm25} indexed, {n_vec} vectors, store={vector_store_name()})"


# ---------- Search / Ranking Helpers for M2-A ----------
def bm25_search(query_text: str, candidate_ids: Optional[list[str]] = None, top_k: int = 50) -> list[tuple[str, float]]:
    """Scores candidate products using BM25Okapi + term frequency matching.
    Returns list of (product_id, score) in descending order.
    """
    tokens = tokenize(query_text)
    if not tokens or _bm25 is None or not _bm25_ids:
        return []

    with _lock:
        raw_scores = _bm25.get_scores(tokens)
        bm25_map = {}
        for idx, pid in enumerate(_bm25_ids):
            doc_tokens = _tokenized_corpus[idx]
            overlap = sum(doc_tokens.count(t) for t in tokens)
            if overlap > 0:
                # BM25 score + small term-overlap bonus to prevent zero-IDF drops on small test sets
                bm25_map[pid] = float(raw_scores[idx]) + 0.1 * overlap

    candidates = candidate_ids if candidate_ids is not None else _bm25_ids
    scored = [(pid, bm25_map[pid]) for pid in candidates if pid in bm25_map]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def dense_search(query_or_vec: Union[str, np.ndarray], candidate_ids: Optional[list[str]] = None,
                 top_k: int = 50) -> list[tuple[str, float]]:
    """Scores candidate products using cosine similarity of multilingual embeddings.
    Returns list of (product_id, score) in descending order.
    """
    with _lock:
        if _vectors is None or not _vector_ids:
            return []
        mat = _vectors
        ids = list(_vector_ids)

    if isinstance(query_or_vec, str):
        q_vec = embeddings.embed_query(query_or_vec)
    else:
        q_vec = np.asarray(query_or_vec, dtype=np.float32)

    if q_vec is None:
        return []

    col = _chroma()
    id_sim: dict[str, float] = {}
    if col is not None:
        # Chroma is the vector store: ask it (cosine space, so similarity = 1 - distance)
        try:
            res = col.query(query_embeddings=[np.asarray(q_vec).tolist()], n_results=max(1, col.count()))
            active = set(ids)
            id_sim = {pid: 1.0 - float(d) for pid, d in zip(res["ids"][0], res["distances"][0]) if pid in active}
        except Exception as exc:
            log.warning("Chroma query failed, using the in-memory copy: %s", exc)
    if not id_sim:
        # Cosine similarity (vectors are L2-normalized)
        sims = np.dot(mat, q_vec)
        id_sim = dict(zip(ids, sims))

    candidates = candidate_ids if candidate_ids is not None else ids
    scored = [(pid, float(id_sim.get(pid, 0.0))) for pid in candidates if pid in id_sim]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# ---------- Adapters used by pipeline/retrieval.py (M2-A) ----------
# Merge note: M2-A's hybrid ranker calls these. They reuse the M2-B index above, so there is one
# BM25 index and one embedding matrix in the process.
def get_bm25_scores(query_tokens: list[str]) -> dict[str, float]:
    """BM25 score per product for already-tokenised query terms (only products that match a term).
    Adds the same small term-overlap bonus as bm25_search(), so common terms never score zero."""
    with _lock:
        if _bm25 is None or not query_tokens or not _bm25_ids:
            return {}
        raw = _bm25.get_scores(query_tokens)
        out: dict[str, float] = {}
        for idx, pid in enumerate(_bm25_ids):
            overlap = sum(_tokenized_corpus[idx].count(t) for t in query_tokens)
            if overlap > 0:
                out[pid] = float(raw[idx]) + 0.1 * overlap
        return out


def get_dense_scores(query_text: str) -> dict[str, float]:
    """Cosine similarity per product (only positive scores); empty when embeddings are disabled."""
    return {pid: s for pid, s in dense_search(query_text, top_k=len(_vector_ids) or 1) if s > 0}


def get_product(product_id: str) -> Optional[Product]:
    with _lock:
        return _products.get(product_id)
