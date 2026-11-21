"""Step 6: no matches -> drop one constraint at a time and say which one blocked results. Owner: M1.

Order: colour, size, brand, then budget raised by up to 25%.
TODO(M1): counterfactual hint ("raise your budget by ₹500 and 4 more options appear").
"""
from __future__ import annotations

from typing import Any, Optional

from app.pipeline import filters, retrieval
from app.schemas import Product

_ORDER = ["color", "size", "brands_include", "budget_max"]
_LABEL = {"color": "colour", "size": "size", "brands_include": "brand", "budget_max": "budget"}


def relax(f: dict[str, Any], query: str) -> Optional[tuple[str, list[tuple[Product, float]]]]:
    """Returns (blocking_constraint_label, ranked_nearest) or None if nothing helps."""
    for key in _ORDER:
        if not f.get(key):
            continue
        loosened = dict(f)
        if key == "budget_max":
            loosened[key] = int(f[key] * 1.25)
        else:
            loosened.pop(key)
        ranked = retrieval.rank(query, filters.apply(loosened), loosened.get("sort"))
        if ranked:
            return _LABEL[key], ranked
    return None
