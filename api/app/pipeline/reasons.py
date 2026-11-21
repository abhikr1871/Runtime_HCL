"""Step 7a: build each product's reason from database fields (the LLM never invents these). Owner: M1 (Abhijeet, Daksh).

Reasons are written in the shopper's language (English, Hindi or Hinglish). Numbers are always
formatted by code, so the validator can check every ₹ amount and stock figure.
Tags the shopper actually asked for ("waterproof", "for trekking") are added when every word of the tag
appears in the request (Daksh), so "shoes for running" never claims "for flat feet".
"""
from __future__ import annotations

import re
from typing import Any, Optional

from app.schemas import Product

_TEXT = {
    "en": {"under": "{price}, under your {budget} budget", "rated": "rated {r:.1f}", "in_stock": "in stock",
           "left": "only {n} left", "brand": "{brand} as you asked", "match": "matches your request",
           "for": "for {tag}"},
    "hinglish": {"under": "{price}, aapke {budget} budget ke andar", "rated": "rating {r:.1f}", "in_stock": "stock mein hai",
                 "left": "sirf {n} bache hain", "brand": "{brand}, jaisa aapne kaha", "match": "aapki request se match karta hai",
                 "for": "{tag} ke liye"},
    "hi": {"under": "{price}, आपके {budget} बजट के अंदर", "rated": "रेटिंग {r:.1f}", "in_stock": "स्टॉक में है",
           "left": "सिर्फ़ {n} बचे हैं", "brand": "{brand}, जैसा आपने कहा", "match": "आपकी ज़रूरत से मेल खाता है",
           "for": "{tag} के लिए"},
}
_GENERIC_TAGS = {"shoes", "shoe", "laptop", "laptops", "phone", "phones", "shirt", "shirts", "bag", "bags", "product", "item"}
_FILLER_WORDS = {"for", "and", "the", "with", "in", "of", "to", "a", "an"}
_USE_TAGS = {"trekking", "hiking", "running", "study", "office", "gaming", "travel", "party", "flat feet", "jogging", "gym"}
_WORD = re.compile(r"[\wऀ-ॿ]+")


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


def _tag_matches_query(tag: str, shopper_words: set[str]) -> bool:
    content = [w for w in _WORD.findall(tag.lower()) if w not in _FILLER_WORDS]
    return bool(content) and all(w in shopper_words for w in content)


def _matched_tags(p: Product, f: dict[str, Any]) -> list[str]:
    """Product tags / attribute values whose every word the shopper used (Daksh)."""
    shopper_words = set(_WORD.findall((f.get("query_text") or "").lower()))
    for attr in f.get("attributes", []) or []:
        shopper_words.update(_WORD.findall(str(attr).lower()))
    values: list[str] = list(p.tags or [])
    for v in (p.attributes or {}).values():
        values.extend([v] if isinstance(v, str) else [x for x in v if isinstance(x, str)] if isinstance(v, list) else [])
    matched: list[str] = []
    for value in values:
        low = value.lower().strip()
        if low not in _GENERIC_TAGS and low not in matched and _tag_matches_query(low, shopper_words):
            matched.append(low)
    return matched


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
    tags = []
    for tag in _matched_tags(p, f)[:2]:
        clean = " ".join(w for w in _WORD.findall(tag) if w not in _FILLER_WORDS)
        tags.append(t["for"].format(tag=clean) if clean in _USE_TAGS else clean)
    if tags:
        parts.extend(tags)
        fields.append("tags")
    return ", ".join(parts) or t["match"], fields
