"""Step 7a: build each product's reason from database fields (the LLM never invents these). Owner: M1."""
from __future__ import annotations

import re
from typing import Any, Optional

from app.schemas import Product

_GENERIC_TAGS = {"shoes", "shoe", "laptop", "laptops", "phone", "phones", "shirt", "shirts", "bag", "bags", "product", "item"}
_FILLER_WORDS = {"for", "and", "the", "with", "in", "of", "to", "a", "an"}


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
    content_tag_words = [w for w in re.findall(r"[\wऀ-ॿ]+", tag.lower()) if w not in _FILLER_WORDS]
    if not content_tag_words:
        return False
    return all(w in shopper_words for w in content_tag_words)


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

    # Match shopper's query words/slots against the product's tags/attributes
    query_text = (f.get("query_text") or "").lower()
    shopper_words = set(re.findall(r"[\wऀ-ॿ]+", query_text))
    for attr in f.get("attributes", []):
        shopper_words.update(re.findall(r"[\wऀ-ॿ]+", attr.lower()))

    matched_tags = []
    for tag in (p.tags or []):
        t_low = tag.lower().strip()
        if t_low in _GENERIC_TAGS:
            continue
        if _tag_matches_query(t_low, shopper_words):
            if t_low not in matched_tags:
                matched_tags.append(t_low)

    for k, v in (p.attributes or {}).items():
        if isinstance(v, str):
            v_low = v.lower().strip()
            if v_low not in _GENERIC_TAGS and _tag_matches_query(v_low, shopper_words):
                if v_low not in matched_tags:
                    matched_tags.append(v_low)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    item_low = item.lower().strip()
                    if item_low not in _GENERIC_TAGS and _tag_matches_query(item_low, shopper_words):
                        if item_low not in matched_tags:
                            matched_tags.append(item_low)

    formatted_tags = []
    for tag in matched_tags[:2]:
        content = [w for w in re.findall(r"[\wऀ-ॿ]+", tag) if w not in _FILLER_WORDS]
        clean_tag = " ".join(content)
        if clean_tag in {"trekking", "hiking", "running", "study", "office", "gaming", "travel", "party", "flat feet", "jogging", "gym"}:
            formatted_tags.append(f"for {clean_tag}")
        else:
            formatted_tags.append(clean_tag)

    if formatted_tags:
        parts.extend(formatted_tags)
        fields.append("tags")

    return ", ".join(parts) or "matches your request", fields
