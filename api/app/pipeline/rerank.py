"""Optional cross-encoder rerank of the top 20 (stretch goal). Owner: M2.

Pass-through unless ENABLE_RERANKER=true. Keep it only if the ablation shows a relevance gain.
"""
from __future__ import annotations

from app.config import get_settings
from app.schemas import Product


def rerank(query: str, ranked: list[tuple[Product, float]], top_n: int = 20) -> list[tuple[Product, float]]:
    if not get_settings().enable_reranker:
        return ranked
    # TODO(M2): score (query, product text) pairs with a small multilingual cross-encoder
    return ranked
