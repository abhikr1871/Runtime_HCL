"""Step 7c: grounding validator. Owner: M1 (Abhijeet, Daksh).

Every ₹ amount, product ID and stock claim in the reply must match the database:
  - ₹ amounts: a shown product's price, or a number the shopper gave (their budget) / the relax hint used
  - product IDs: only products in the result set
  - stock claims are checked against the product named in the SAME sentence (Daksh), in English,
    Hinglish and Hindi: "only 2 left" / "sirf 2 bache" / "सिर्फ़ 2 बचे", "out of stock" / "स्टॉक में नहीं",
    "in stock" / "stock mein hai" / "स्टॉक में है"
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
_N_LEFT = re.compile(r"(?:\b(?:only|sirf)\s+|सिर्फ़\s*|सिर्फ\s*)?(\d+)\s*(?:left\b|bache\b|बचे)", re.I)
_OUT_OF_STOCK = re.compile(r"\bout\s+of\s+stock\b|\bstock\s+(?:mein\s+)?nahi\b|स्टॉक\s*में\s*नहीं", re.I)
_IN_STOCK = re.compile(r"\bin\s+stock\b|\bavailable\b|\bstock\s+mein\s+hai\b|स्टॉक\s*में\s*है", re.I)
_UNAVAILABLE = re.compile(r"\bunavailable\b", re.I)


@dataclass
class ValidationResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


def _segments(answer: str) -> list[str]:
    """Split into sentences/lines; a line naming several products is split per product (Daksh)."""
    segments = []
    for ln in (x.strip() for x in re.split(r"[\n;]", answer)):
        if not ln:
            continue
        if len(_PRODUCT_ID.findall(ln)) > 1:
            parts = re.split(r"(?<=[.!?।])\s+|(?<!^)(?=\bP-\d{4,}\b)", ln)
            segments.extend(p.strip() for p in parts if p.strip())
        else:
            segments.append(ln)
    return segments


def check(answer: str, products: list[Product], allowed_amounts: Iterable[int] = ()) -> ValidationResult:
    """`allowed_amounts` = numbers the shopper gave us (e.g. their budget), which may be repeated back."""
    prices = {p.price_inr for p in products if p.price_inr is not None} | set(allowed_amounts)
    by_id = {p.product_id: p for p in products}
    problems: list[str] = []

    # 1. prices
    for raw in _AMOUNT.findall(answer):
        amount = int(raw.replace(",", ""))
        if amount not in prices:
            problems.append(f"amount ₹{amount} is not a catalogue price or the shopper's budget")
    # 2. product IDs
    for pid in _PRODUCT_ID.findall(answer):
        if pid not in by_id:
            problems.append(f"product {pid} is not in the result set")
    # 3. stock claims, per sentence
    for seg in _segments(answer):
        named = [by_id[pid] for pid in _PRODUCT_ID.findall(seg) if pid in by_id]
        for m in _N_LEFT.finditer(seg):
            n = int(m.group(1))
            if named:
                problems += [f"{p.product_id} claimed {n} left but actual stock is {p.stock}" for p in named if p.stock != n]
            elif not any(p.stock == n for p in products):
                problems.append(f"claimed {n} left but no shown product has stock {n}")
        out = bool(_OUT_OF_STOCK.search(seg))
        if out:
            if named:
                problems += [f"{p.product_id} claimed out of stock but actual stock is {p.stock}" for p in named if p.stock > 0]
            elif products and all(p.stock > 0 for p in products):
                problems.append("claimed out of stock but all shown products are in stock")
        if _IN_STOCK.search(seg) and not out and not _UNAVAILABLE.search(seg):
            if named:
                problems += [f"{p.product_id} claimed in stock but actual stock is 0" for p in named if p.stock <= 0]
            elif products and all(p.stock <= 0 for p in products):
                problems.append("claimed in stock but all shown products are out of stock")
    return ValidationResult(ok=not problems, problems=problems)
