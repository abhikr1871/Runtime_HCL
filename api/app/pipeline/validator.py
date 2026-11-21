"""Step 7c: grounding validator. Every ₹ amount and product ID in the reply must match the database. Owner: M1.

On failure the orchestrator falls back to the code template, so a wrong number never reaches the shopper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from app.schemas import Product

_AMOUNT = re.compile(r"(?:₹|rs\.?\s*|inr\s*)(\d[\d,]*)", re.I)
_PRODUCT_ID = re.compile(r"\bP-\d{4,}\b")


@dataclass
class ValidationResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


def check(answer: str, products: list[Product], allowed_amounts: Iterable[int] = ()) -> ValidationResult:
    """`allowed_amounts` = numbers the shopper gave us (e.g. their budget), which may be repeated back."""
    prices = {p.price_inr for p in products if p.price_inr is not None} | set(allowed_amounts)
    ids = {p.product_id for p in products}
    problems = []
    for raw in _AMOUNT.findall(answer):
        amount = int(raw.replace(",", ""))
        if amount not in prices:
            problems.append(f"amount ₹{amount} is not a catalogue price or the shopper's budget")
    for pid in _PRODUCT_ID.findall(answer):
        if pid not in ids:
            problems.append(f"product {pid} is not in the result set")
    # TODO(M1): stock claims ("in stock", "only N left") must match product.stock
    return ValidationResult(ok=not problems, problems=problems)
