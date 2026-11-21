"""Comparison table built purely from catalogue fields (no LLM). Owner: M3."""
from __future__ import annotations

from app.schemas import ComparisonTable, Product

_COLUMNS = ["product_id", "name", "brand", "price_inr", "rating", "stock", "weight_g"]


def build_table(products: list[Product]) -> ComparisonTable:
    rows = []
    for p in products:
        row = {c: getattr(p, c) for c in _COLUMNS}
        rows.append({k: ("not listed" if v is None else v) for k, v in row.items()})
    # TODO(M3): add category-specific attributes present on all compared products
    return ComparisonTable(columns=_COLUMNS, rows=rows)
