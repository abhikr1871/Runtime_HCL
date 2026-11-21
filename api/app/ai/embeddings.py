"""Multilingual text embeddings (intfloat/multilingual-e5-small). Owner: M2.

e5 needs prefixes: 'query: ' for the shopper's text, 'passage: ' for product text.
Loaded lazily, and only when ENABLE_EMBEDDINGS=true, so the app and tests run without the model.
"""
from __future__ import annotations

import logging

from app.config import get_settings

log = logging.getLogger("embeddings")
_model = None


def get_model():
    global _model
    s = get_settings()
    if not s.enable_embeddings:
        return None
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(s.embed_model, cache_folder=str(s.models_dir))
        except Exception as exc:
            log.warning("embedding model unavailable, falling back to keyword search: %s", exc)
            return None
    return _model


def embed_query(text: str):
    m = get_model()
    return None if m is None else m.encode(f"query: {text}", normalize_embeddings=True)


def embed_passages(texts: list[str]):
    m = get_model()
    return None if m is None else m.encode([f"passage: {t}" for t in texts], normalize_embeddings=True)
