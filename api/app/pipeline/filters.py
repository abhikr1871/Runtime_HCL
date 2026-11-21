"""Step 5a: hard constraints in SQL, BEFORE ranking. Owner: M2.

Anything that breaks the budget, brand or stock constraint is removed here,
so it can never appear in an answer. Only fixed column names + ? placeholders are used.
"""
from __future__ import annotations

from typing import Any

from app.schemas import Product
from app.store import repository


def build_where(f: dict[str, Any], in_stock_only: bool = True) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if in_stock_only:
        clauses.append("stock > 0")
    if f.get("budget_max") is not None:
        clauses.append("price_inr IS NOT NULL AND price_inr <= ?")
        params.append(int(f["budget_max"]))
    if f.get("budget_min") is not None:
        clauses.append("price_inr IS NOT NULL AND price_inr >= ?")
        params.append(int(f["budget_min"]))
    if f.get("brands_include"):
        clauses.append(f"LOWER(brand) IN ({','.join('?' * len(f['brands_include']))})")
        params += [b.lower() for b in f["brands_include"]]
    if f.get("brands_exclude"):
        clauses.append(f"(brand IS NULL OR LOWER(brand) NOT IN ({','.join('?' * len(f['brands_exclude']))}))")
        params += [b.lower() for b in f["brands_exclude"]]
    if f.get("category"):
        clauses.append("category = ?")
        params.append(f["category"])
    # TODO(M2): size and colour via json_each(sizes_json / colors_json)
    return (" AND ".join(clauses) or "1=1"), params


def apply(f: dict[str, Any]) -> list[Product]:
    return repository.find_products(*build_where(f))


def out_of_stock_matches(f: dict[str, Any]) -> list[Product]:
    """Matches that are only blocked by stock - used for 'your usual X is out of stock' substitutes."""
    where, params = build_where(f, in_stock_only=False)
    return repository.find_products(f"({where}) AND stock <= 0", params)
