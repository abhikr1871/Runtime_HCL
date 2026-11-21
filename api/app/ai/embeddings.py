"""Multilingual text embeddings (intfloat/multilingual-e5-small). Owner: M2-B.

e5 needs prefixes:
- 'query: ' for the shopper's query text
- 'passage: ' for product document text

Loaded lazily, and only when ENABLE_EMBEDDINGS=true, so the app and tests run without the model.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np

from app.config import get_settings
from app.schemas import Product

log = logging.getLogger("embeddings")
_model = None


def get_model():
    """Lazily load sentence-transformers model when enabled."""
    global _model
    s = get_settings()
    if not s.enable_embeddings:
        return None
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            models_dir = str(s.models_dir) if s.models_dir.exists() else None
            _model = SentenceTransformer(s.embed_model, cache_folder=models_dir)
            log.info("loaded multilingual embedding model %s", s.embed_model)
        except Exception as exc:
            log.warning("embedding model unavailable, falling back to keyword search: %s", exc)
            return None
    return _model


def prepare_product_text(p: Union[Product, dict[str, Any]]) -> str:
    """Prepares clean, rich text representation for embedding passage generation.
    Supports English, Hindi, and Hinglish products.
    """
    if isinstance(p, Product):
        name = p.name
        brand = p.brand
        category = p.category
        subcategory = p.subcategory
        description = p.description
        tags = p.tags
    else:
        name = p.get("name", "")
        brand = p.get("brand")
        category = p.get("category", "")
        subcategory = p.get("subcategory")
        description = p.get("description", "")
        tags = p.get("tags") or []

    parts = [name]
    if brand:
        parts.append(f"Brand: {brand}")
    if category:
        parts.append(f"Category: {category}")
    if subcategory:
        parts.append(f"Type: {subcategory}")
    if tags:
        parts.append("Tags: " + " ".join(tags))
    if description:
        parts.append(description)

    return ". ".join(p.strip().rstrip(".") for p in parts if p and str(p).strip())


def embed_query(text: str) -> Optional[np.ndarray]:
    """Generates L2-normalized embedding vector for shopper query with 'query: ' prefix."""
    m = get_model()
    if m is None or not text.strip():
        return None
    try:
        vec = m.encode(f"query: {text.strip()}", normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)
    except Exception as exc:
        log.warning("failed to embed query: %s", exc)
        return None


def embed_passages(texts: list[str], batch_size: int = 32) -> Optional[np.ndarray]:
    """Generates L2-normalized embedding matrix for product texts with 'passage: ' prefix."""
    m = get_model()
    if m is None or not texts:
        return None
    try:
        passages = [f"passage: {t.strip()}" for t in texts]
        vecs = m.encode(passages, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    except Exception as exc:
        log.warning("failed to embed passages: %s", exc)
        return None


def vector_to_bytes(vec: np.ndarray) -> bytes:
    """Serialize float32 numpy vector to raw bytes for SQLite BLOB storage."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def bytes_to_vector(b: bytes) -> np.ndarray:
    """Deserialize raw bytes from SQLite BLOB to float32 numpy vector."""
    return np.frombuffer(b, dtype=np.float32)
