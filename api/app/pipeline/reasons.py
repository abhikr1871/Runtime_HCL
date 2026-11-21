"""Step 7a: build each product's reason from database fields (the LLM never invents these). Owner: M1.

Reasons are written in the shopper's language (English, Hindi or Hinglish). Numbers are always
formatted by code, so the validator can check every ₹ amount and stock figure.
"""
from __future__ import annotations

from typing import Any, Optional

from app.schemas import Product

_TEXT = {
    "en": {"under": "{price}, under your {budget} budget", "rated": "rated {r:.1f}", "in_stock": "in stock",
           "left": "only {n} left", "brand": "{brand} as you asked", "match": "matches your request"},
    "hinglish": {"under": "{price}, aapke {budget} budget ke andar", "rated": "rating {r:.1f}", "in_stock": "stock mein hai",
                 "left": "sirf {n} bache hain", "brand": "{brand}, jaisa aapne kaha", "match": "aapki request se match karta hai"},
    "hi": {"under": "{price}, आपके {budget} बजट के अंदर", "rated": "रेटिंग {r:.1f}", "in_stock": "स्टॉक में है",
           "left": "सिर्फ़ {n} बचे हैं", "brand": "{brand}, जैसा आपने कहा", "match": "आपकी ज़रूरत से मेल खाता है"},
}


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


def build(p: Product, f: dict[str, Any], language: str = "en") -> tuple[str, list[str]]:
    """Returns (reason sentence in the shopper's language, catalogue fields used)."""
    t = _TEXT.get(language, _TEXT["en"])
    parts, fields = [], []
    if p.price_inr is not None:
        budget = f.get("budget_max")
        parts.append(t["under"].format(price=format_inr(p.price_inr), budget=format_inr(budget))
                     if budget and p.price_inr <= budget else format_inr(p.price_inr))
        fields.append("price")
    if p.rating is not None:
        parts.append(t["rated"].format(r=p.rating))
        fields.append("rating")
    if p.stock > 0:
        parts.append(t["in_stock"] if p.stock > 3 else t["left"].format(n=p.stock))
        fields.append("stock")
    if p.brand and p.brand in f.get("brands_include", []):
        parts.append(t["brand"].format(brand=p.brand))
        fields.append("brand")
    # TODO(M1, stretch): matched tags ("waterproof", "for flat feet") and review highlights cited by review ID
    return ", ".join(parts) or t["match"], fields
