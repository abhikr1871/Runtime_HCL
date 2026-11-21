"""Optional cross-encoder rerank of the top 20 (stretch goal). Owner: M2.

Pass-through unless ENABLE_RERANKER=true. Keep it only if the ablation shows a relevance gain.
Provides a graceful fallback if the reranker model is unavailable.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.config import get_settings
from app.schemas import Product

log = logging.getLogger("rerank")
_cross_encoder = None
_model_failed = False


def _get_reranker():
    global _cross_encoder, _model_failed
    s = get_settings()
    if not s.enable_reranker or _model_failed:
        return None
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(s.reranker_model)
        except Exception as exc:
            log.warning("Cross-encoder model unavailable, falling back to hybrid ranking: %s", exc)
            _model_failed = True
            return None
    return _cross_encoder


def rerank(
    query: str,
    ranked: list[tuple[Product, float]],
    top_n: int = 20,
) -> list[tuple[Product, float]]:
    """Rerank top_n candidates using a cross-encoder model when ENABLE_RERANKER is enabled."""
    if not get_settings().enable_reranker or not ranked or not query.strip():
        return ranked

    encoder = _get_reranker()
    if encoder is None:
        return ranked

    try:
        candidates = ranked[:top_n]
        remaining = ranked[top_n:]

        pairs = [
            (query, f"{p.name} {p.brand or ''} {p.category} {p.description}")
            for p, _ in candidates
        ]
        scores = encoder.predict(pairs)

        reranked_top = sorted(
            [(p, float(s)) for (p, _), s in zip(candidates, scores)],
            key=lambda x: -x[1],
        )
        return reranked_top + remaining
    except Exception as exc:
        log.warning("Cross-encoder reranking failed: %s", exc)
        return ranked
