"""Step 7c: grounding validator. Owner: M1.

Every ₹ amount, product ID and stock claim in the reply must match the database:
  - ₹ amounts: a shown product's price, or a number the shopper gave (their budget)
  - product IDs: only products in the result set
  - "only N left" / "सिर्फ़ N बचे" / "sirf N bache": N must be a shown product's stock
  - "out of stock" claims: at least one shown product must really be out of stock
On failure the orchestrator regenerates once, then falls back to the code template,
so a wrong number never reaches the shopper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from app.schemas import Product

_AMOUNT = re.compile(r"(?:₹|rs\.?\s*|inr\s*)(\d[\d,]*)", re.I)
_PRODUCT_ID = re.compile(r"\bP-\d{4,}\b")
_ONLY_LEFT = re.compile(r"(?:only|sirf|सिर्फ़|सिर्फ)\s*(\d+)\s*(?:left|bache|बचे|units?)", re.I)
_OUT_OF_STOCK = re.compile(r"out of stock|stock (?:mein )?nahi|स्टॉक में नहीं", re.I)


@dataclass
class ValidationResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


def check(answer: str, products: list[Product], allowed_amounts: Iterable[int] = ()) -> ValidationResult:
    """`allowed_amounts` = numbers the shopper gave us (e.g. their budget), which may be repeated back."""
    prices = {p.price_inr for p in products if p.price_inr is not None} | set(allowed_amounts)
    ids = {p.product_id for p in products}
    stocks = {p.stock for p in products}
    problems = []
    for raw in _AMOUNT.findall(answer):
        amount = int(raw.replace(",", ""))
        if amount not in prices:
            problems.append(f"amount ₹{amount} is not a catalogue price or the shopper's budget")
    for pid in _PRODUCT_ID.findall(answer):
        if pid not in ids:
            problems.append(f"product {pid} is not in the result set")
    for n in _ONLY_LEFT.findall(answer):
        if int(n) not in stocks:
            problems.append(f"'only {n} left' does not match any shown product's stock")
    if _OUT_OF_STOCK.search(answer) and all(p.stock > 0 for p in products):
        problems.append("claims something is out of stock, but every shown product is in stock")
    return ValidationResult(ok=not problems, problems=problems)
