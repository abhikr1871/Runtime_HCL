"""Step 4: decide whether to ask ONE clarifying question (max 2 in a row). Owner: M1 (Abhijeet, Daksh).

WHEN to ask (Abhijeet) - only when the request is genuinely broad, e.g. "I need a laptop", "need earbuds":
  - exactly one product word and nothing else specific, AND
  - either it is a broad item (laptop, phone, shoes, watch, gift...) or many products match it, AND
  - more than 3 products match (with 3 or fewer, just show them), AND
  - the shopper has not given a budget or brand yet.
Anything more specific ("wireless earbuds", "running shoes not Nike", "kurta under 1500") is searched directly:
a useless question costs the shopper a turn, and the brief penalises unnecessary follow-ups.

WHAT to ask (Daksh) - the detail whose values vary most among the current matches (size, use, budget,
brand), with real example options from those matches. If nothing varies, fall back to asking the budget.
A detail is never asked twice in one conversation.
"""
from __future__ import annotations

import re
from typing import Optional

from app.pipeline.reasons import format_inr
from app.schemas import Product, SessionState

MAX_STREAK = 2
MIN_MATCHES_TO_ASK = 3      # with 3 or fewer matches, show them instead of asking
MANY_MATCHES = 8            # a non-broad word is still "broad" when this many products match
BROAD_ITEMS = {"laptop", "laptops", "phone", "phones", "smartphone", "smartphones", "mobile", "shoes", "shoe",
               "watch", "watches", "gift", "gifts", "tablet", "tablets", "clothes", "dress", "bag", "bags"}
_FILLER = {
    "i", "need", "want", "a", "an", "the", "some", "me", "show", "for", "please", "looking", "buy", "get", "find",
    "suggest", "recommend", "something", "good", "nice", "new", "any", "my", "to", "is", "can", "you", "with",
    "in", "best", "give", "mujhe", "chahiye", "ek", "koi", "dikhao", "dikhaye", "do", "kuch", "chaiye", "chahie",
}
_BLOCKING = ("budget_max", "budget_min", "brands_include", "brands_exclude")

_CATEGORY_ATTRIBUTES = {
    "mens-shoes": ["size", "use", "budget_max", "brand"],
    "womens-shoes": ["size", "use", "budget_max", "brand"],
    "shoes": ["size", "use", "budget_max", "brand"],
    "footwear": ["size", "use", "budget_max", "brand"],
    "laptops": ["use", "budget_max", "brand"],
    "mobile-accessories": ["use", "budget_max", "brand"],
    "mens-shirts": ["size", "budget_max", "brand"],
    "clothing": ["size", "budget_max", "brand"],
    "bags": ["use", "budget_max", "brand"],
    "sports-accessories": ["use", "budget_max", "brand"],
}
_DEFAULT_ATTRIBUTES = ["budget_max", "use", "size", "brand"]
_GENERIC_TAGS = {"shoes", "shoe", "laptop", "laptops", "phone", "phones", "shirt", "shirts", "bag", "bags", "gift"}

_QUESTIONS = {
    "size": {
        "en": "What size are you looking for? (e.g. {options})",
        "hinglish": "Aapko kaun sa size chahiye? (jaise {options})",
        "hi": "आपको कौन सा साइज़ चाहिए? (जैसे {options})",
    },
    "use": {
        "en": "What will you be using it for? (e.g. {options})",
        "hinglish": "Aap ise kis use ke liye dhoondh rahe hain? (jaise {options})",
        "hi": "आप इसे किस काम के लिए ढूँढ रहे हैं? (जैसे {options})",
    },
    "budget_max": {
        "en": "What's your budget? (e.g. {options})",
        "hinglish": "Aapka budget kitna hai? (jaise {options})",
        "hi": "आपका बजट कितना है? (जैसे {options})",
    },
    "brand": {
        "en": "Do you have a preferred brand? (e.g. {options})",
        "hinglish": "Aapki koi pasandida brand hai? (jaise {options})",
        "hi": "क्या आपकी कोई पसंदीदा ब्रांड है? (जैसे {options})",
    },
}
_DEFAULT_BUDGET_QUESTION = {
    "en": "What's your budget? For example, under ₹2,000 or under ₹5,000.",
    "hinglish": "Aapka budget kitna hai? Jaise ₹2,000 se kam ya ₹5,000 se kam.",
    "hi": "आपका बजट कितना है? जैसे ₹2,000 से कम या ₹5,000 से कम।",
}


def product_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[\wऀ-ॿ]+", text.lower()) if w not in _FILLER and not w.isdigit()]


def _attribute_variance(attr: str, matches: list[Product], language: str = "en") -> tuple[int, str]:
    """(number of distinct values among the matches, 2-4 example options). Daksh."""
    if attr == "size":
        distinct = sorted({s for p in matches for s in (p.sizes or [])}, key=lambda x: (len(x), x))
        return (len(distinct), ", ".join(distinct[:4])) if len(distinct) >= 2 else (0, "")
    if attr == "use":
        distinct = sorted({t.lower().strip() for p in matches for t in (p.tags or [])} - _GENERIC_TAGS)
        return (len(distinct), ", ".join(distinct[:3])) if len(distinct) >= 2 else (0, "")
    if attr == "brand":
        distinct = sorted({p.brand for p in matches if p.brand})
        return (len(distinct), ", ".join(distinct[:3])) if len(distinct) >= 2 else (0, "")
    if attr == "budget_max":
        prices = [p.price_inr for p in matches if p.price_inr is not None]
        if len(set(prices)) < 2 or max(prices) - min(prices) < 500:
            return 0, ""
        low, high = format_inr(int(round(min(prices) * 1.5, -2))), format_inr(int(round(max(prices) * 0.9, -2)))
        options = {"hi": f"{low} से कम या {high} से कम", "hinglish": f"{low} se kam ya {high} se kam"}.get(
            language, f"under {low} or under {high}")
        return len(set(prices)), options
    return 0, ""


def _wide_price_spread(matches: list[Product]) -> bool:
    prices = [p.price_inr for p in matches if p.price_inr]
    return len(prices) >= 2 and max(prices) >= 1.5 * min(prices)


def next_question(state: SessionState, matches: list[Product], language: str) -> Optional[tuple[str, str]]:
    """`matches` = products that matched the request after filters. Returns (slot, question) or None."""
    f = state.filters
    if state.clarify_streak >= MAX_STREAK or not matches:
        return None
    if any(f.get(k) for k in _BLOCKING):
        return None
    words = product_words(f.get("query_text", ""))
    if len(words) != 1:
        return None
    if words[0] not in BROAD_ITEMS and len(matches) <= MANY_MATCHES:
        return None
    # 3 or fewer matches: just show them - unless their prices are far apart (₹38,990 vs ₹72,990 laptops),
    # where the budget decides which one fits, so that is the only question worth asking
    few = len(matches) <= MIN_MATCHES_TO_ASK
    if few and not _wide_price_spread(matches):
        return None

    # which detail to ask: the one that varies most among the matches (Daksh)
    cat = f.get("category") or (matches[0].category if len({p.category for p in matches}) == 1 else None)
    best_attr, best_var, best_opts = None, 0, ""
    for attr in ["budget_max"] if few else _CATEGORY_ATTRIBUTES.get(cat, _DEFAULT_ATTRIBUTES):
        if f.get(attr) or attr in state.asked_slots:
            continue
        var, opts = _attribute_variance(attr, matches, language)
        if var > best_var:
            best_attr, best_var, best_opts = attr, var, opts
    if best_attr:
        texts = _QUESTIONS[best_attr]
        return best_attr, texts.get(language, texts["en"]).format(options=best_opts)

    # nothing varies enough: ask the budget once
    if "budget_max" not in state.asked_slots:
        return "budget_max", _DEFAULT_BUDGET_QUESTION.get(language, _DEFAULT_BUDGET_QUESTION["en"])
    return None
