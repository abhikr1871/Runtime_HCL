"""Step 5a: hard constraints in SQL, BEFORE ranking. Owner: M2.

Anything that breaks the budget, brand, stock, size, colour or category constraint is removed here,
so it can never appear in an answer. Only fixed column names + ? placeholders are used.
"""
from __future__ import annotations

from typing import Any

from app.schemas import Product
from app.store import repository


def _clean_str(val: Any) -> str:
    return str(val).strip().lower()


def build_where(f: dict[str, Any], in_stock_only: bool = True) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    # 1. Stock / Availability constraint
    if f.get("in_stock") is False:
        clauses.append("stock <= 0")
    elif in_stock_only or f.get("in_stock") is True:
        clauses.append("stock > 0")

    # 2. Maximum Budget constraint
    if f.get("budget_max") is not None:
        clauses.append("price_inr IS NOT NULL AND price_inr <= ?")
        params.append(int(f["budget_max"]))

    # 3. Minimum Budget constraint
    if f.get("budget_min") is not None:
        clauses.append("price_inr IS NOT NULL AND price_inr >= ?")
        params.append(int(f["budget_min"]))

    # 4. Included Brands constraint
    if f.get("brands_include"):
        include = [b.lower() for b in f["brands_include"] if b]
        if include:
            clauses.append(f"brand IS NOT NULL AND LOWER(brand) IN ({','.join('?' * len(include))})")
            params.extend(include)

    # 5. Excluded Brands constraint
    if f.get("brands_exclude"):
        exclude = [b.lower() for b in f["brands_exclude"] if b]
        if exclude:
            clauses.append(f"(brand IS NULL OR LOWER(brand) NOT IN ({','.join('?' * len(exclude))}))")
            params.extend(exclude)

    # 6. Category / Subcategory / Product Type constraint
    cat = f.get("category") or f.get("product_type")
    if cat:
        c_clean = _clean_str(cat)
        clauses.append("(LOWER(category) = ? OR LOWER(COALESCE(subcategory, '')) = ?)")
        params.extend([c_clean, c_clean])

    # 7. Size constraint (using SQLite json_each on sizes_json)
    if f.get("size"):
        sz = _clean_str(f["size"])
        uk_sz = f"uk {sz}" if not sz.startswith("uk") else sz
        clauses.append(
            "("
            "EXISTS ("
            "  SELECT 1 FROM json_each(sizes_json) "
            "  WHERE LOWER(value) = ? OR LOWER(value) = ? OR LOWER(value) LIKE ?"
            ") OR ("
            "  sizes_json = '[]' AND ("
            "    LOWER(name) LIKE ? OR LOWER(description) LIKE ?"
            "  )"
            ")"
            ")"
        )
        params.extend([sz, uk_sz, f"%{sz}%", f"%size {sz}%", f"%size {sz}%"])

    # 8. Colour constraint (using SQLite json_each on colors_json)
    if f.get("color"):
        col = _clean_str(f["color"])
        clauses.append(
            "("
            "EXISTS ("
            "  SELECT 1 FROM json_each(colors_json) "
            "  WHERE LOWER(value) = ?"
            ") OR ("
            "  colors_json = '[]' AND ("
            "    LOWER(name) LIKE ? OR LOWER(description) LIKE ?"
            "  )"
            ")"
            ")"
        )
        params.extend([col, f"%{col}%", f"%{col}%"])

    # 9. Minimum rating constraint
    if f.get("rating_min") is not None:
        clauses.append("rating IS NOT NULL AND rating >= ?")
        params.append(float(f["rating_min"]))

    return (" AND ".join(clauses) or "1=1"), params


def apply(f: dict[str, Any]) -> list[Product]:
    return repository.find_products(*build_where(f))


def out_of_stock_matches(f: dict[str, Any]) -> list[Product]:
    """Matches that are only blocked by stock - used for 'your usual X is out of stock' substitutes."""
    where, params = build_where(f, in_stock_only=False)
    return repository.find_products(f"({where}) AND stock <= 0", params)
