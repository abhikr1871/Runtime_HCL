"""Step 7a: build each product's reason from database fields (the LLM never invents these). Owner: M1."""
from __future__ import annotations

from typing import Any, Optional

from app.schemas import Product


def format_inr(amount: Optional[int]) -> str:
    """Indian digit grouping: 120000 -> '₹1,20,000'."""
    if amount is None:
        return "price not listed"
    s = str(int(amount))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    return f"₹{s}"


def build(p: Product, f: dict[str, Any]) -> tuple[str, list[str]]:
    """Returns (reason sentence, catalogue fields used)."""
    parts, fields = [], []
    if p.price_inr is not None:
        budget = f.get("budget_max")
        parts.append(f"{format_inr(p.price_inr)}, under your {format_inr(budget)} budget" if budget else format_inr(p.price_inr))
        fields.append("price")
    if p.rating is not None:
        parts.append(f"rated {p.rating:.1f}")
        fields.append("rating")
    if p.stock > 0:
        parts.append("in stock" if p.stock > 3 else f"only {p.stock} left")
        fields.append("stock")
    if p.brand and p.brand in f.get("brands_include", []):
        parts.append(f"{p.brand} as you asked")
        fields.append("brand")
    # TODO(M1): matched attributes/tags ("waterproof", "for trekking") and review highlights
    return ", ".join(parts) or "matches your request", fields
