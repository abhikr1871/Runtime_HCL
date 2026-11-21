"""Step 7c: grounding validator. Every ₹ amount, product ID and stock claim must match the database. Owner: M1.

On failure the orchestrator regenerates once, and on repeated failure falls back to the code template.
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
    prod_map = {p.product_id: p for p in products}
    problems = []

    # 1. Price checks
    for raw in _AMOUNT.findall(answer):
        amount = int(raw.replace(",", ""))
        if amount not in prices:
            problems.append(f"amount ₹{amount} is not a catalogue price or the shopper's budget")

    # 2. Product ID checks
    for pid in _PRODUCT_ID.findall(answer):
        if pid not in ids:
            problems.append(f"product {pid} is not in the result set")

    # 3. Stock claim checks
    lines = [ln.strip() for ln in re.split(r"[\n;]", answer) if ln.strip()]
    segments = []
    for ln in lines:
        pids = _PRODUCT_ID.findall(ln)
        if len(pids) > 1:
            sub = re.split(r"(?<=[.!?])\s+|(?<!^)(?=\bP-\d{4,}\b)", ln)
            segments.extend([sub_s.strip() for sub_s in sub if sub_s.strip()])
        else:
            segments.append(ln)

    for seg in segments:
        seg_pids = _PRODUCT_ID.findall(seg)
        seg_products = [prod_map[pid] for pid in seg_pids if pid in prod_map]

        # "only N left" / "N left"
        for m in re.finditer(r"\b(?:only\s+)?(\d+)\s+left\b", seg, re.I):
            n = int(m.group(1))
            if seg_products:
                for p in seg_products:
                    if p.stock != n:
                        problems.append(f"{p.product_id} claimed {n} left but actual stock is {p.stock}")
            else:
                if not any(p.stock == n for p in products):
                    problems.append(f"claimed {n} left but no shown product has stock {n}")

        # "out of stock"
        if re.search(r"\bout\s+of\s+stock\b", seg, re.I):
            if seg_products:
                for p in seg_products:
                    if p.stock > 0:
                        problems.append(f"{p.product_id} claimed out of stock but actual stock is {p.stock}")
            else:
                if products and all(p.stock > 0 for p in products):
                    problems.append("claimed out of stock but all shown products are in stock")

        # "in stock" / "available"
        if re.search(r"\b(?:in\s+stock|available)\b", seg, re.I) and not re.search(r"\bout\s+of\s+stock\b", seg, re.I) and not re.search(r"\bunavailable\b", seg, re.I):
            if seg_products:
                for p in seg_products:
                    if p.stock == 0:
                        problems.append(f"{p.product_id} claimed in stock but actual stock is 0")
            else:
                if products and all(p.stock == 0 for p in products):
                    problems.append("claimed in stock but all shown products are out of stock")

    return ValidationResult(ok=not problems, problems=problems)
