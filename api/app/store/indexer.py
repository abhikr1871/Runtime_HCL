"""In-memory search index, refreshed after /ingest. Owner: M2.

v0 keeps the active product list in memory and ranks by keyword overlap (pipeline/retrieval.py).
TODO(M2):
  - BM25 index over name + brand + tags + description (rank_bm25)
  - e5 embeddings ('passage: ' prefix) stored in the `embeddings` table, loaded into a NumPy matrix
  - refresh(product_ids) re-embeds only changed rows and swaps the index atomically
"""
from __future__ import annotations

import threading

from app.schemas import Product
from app.store import repository

_lock = threading.Lock()
_products: dict[str, Product] = {}


def rebuild() -> int:
    global _products
    fresh = {p.product_id: p for p in repository.find_products()}
    with _lock:
        _products = fresh
    return len(fresh)


def refresh(product_ids: list[str]) -> None:
    """Called by /ingest so new or changed products are searchable on the next request."""
    for p in repository.get_products(product_ids):
        with _lock:
            _products[p.product_id] = p


def status() -> str:
    return f"keyword-v0 ({len(_products)} products)"
