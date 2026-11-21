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
from collections import Counter
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
    "options", "option", "products", "product", "items", "item", "ideas", "suggestions", "stuff", "of", "few",
    "dikhaiye", "batao", "bataiye",
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


_USE_TAGS = {"running", "trekking", "hiking", "walking", "jogging", "gym", "fitness", "sports", "yoga", "cricket",
             "football", "office", "work", "formal", "party", "casual", "wedding", "festive", "travel", "daily",
             "outdoor", "study", "school", "college", "gaming", "music", "kids", "gift"}


def _round_up(x: int) -> int:
    """Round up to 2 significant digits, at least to the hundred: 2,412 -> 2,500; 1,36,880 -> 1,40,000."""
    step = max(10 ** (len(str(int(x))) - 2), 100)
    return int(-(-x // step) * step)


def product_words(text: str) -> list[str]:
    from app.pipeline.retrieval import _filler_typo   # "optons" is a typo of "options", not a product
    return [w for w in re.findall(r"[\wऀ-ॿ]+", text.lower()) if w not in _FILLER and not w.isdigit() and not _filler_typo(w)]


def _attribute_variance(attr: str, matches: list[Product], language: str = "en") -> tuple[int, str]:
    """(number of distinct values among the matches, 2-4 example options). Daksh."""
    if attr == "size":
        distinct = sorted({s for p in matches for s in (p.sizes or [])}, key=lambda x: (len(x), x))
        return (len(distinct), ", ".join(distinct[:4])) if len(distinct) >= 2 else (0, "")
    if attr == "use":
        # only real uses ("running", "office"), most common first - never adjectives like "agile"
        counts = Counter(t.lower().strip() for p in matches for t in (p.tags or []) if t.lower().strip() in _USE_TAGS)
        distinct = [t for t, _ in counts.most_common()]
        return (len(distinct), ", ".join(distinct[:3])) if len(distinct) >= 2 else (0, "")
    if attr == "brand":
        distinct = sorted({p.brand for p in matches if p.brand})
        return (len(distinct), ", ".join(distinct[:3])) if len(distinct) >= 2 else (0, "")
    if attr == "budget_max":
        prices = sorted(p.price_inr for p in matches if p.price_inr is not None)
        if len(set(prices)) < 2 or prices[-1] - prices[0] < 500:
            return 0, ""
        # round numbers that split the matches into cheaper / middle / dearer thirds
        lo_n = _round_up(prices[len(prices) // 3])
        hi_n = _round_up(prices[(2 * len(prices)) // 3])
        if hi_n <= lo_n:
            hi_n = _round_up(prices[-1])
        low, high = format_inr(lo_n), format_inr(hi_n)
        options = {"hi": f"{low} से कम या {high} से कम", "hinglish": f"{low} se kam ya {high} se kam"}.get(
            language, f"under {low} or under {high}")
        return len(set(prices)), options
    return 0, ""


def _wide_price_spread(matches: list[Product]) -> bool:
    prices = [p.price_inr for p in matches if p.price_inr]
    return len(prices) >= 2 and max(prices) >= 1.5 * min(prices)


_SHOW_ALL = {"en": "Show all", "hinglish": "Sab dikhao", "hi": "सब दिखाओ"}
WHAT_ARE_YOU_LOOKING_FOR = {
    "en": "Happy to help! What are you looking for? For example shoes, a laptop, earbuds or a gift.",
    "hinglish": "Zaroor! Aap kya dhoondh rahe hain? Jaise shoes, laptop, earbuds ya koi gift.",
    "hi": "ज़रूर! आप क्या ढूँढ रहे हैं? जैसे जूते, लैपटॉप, ईयरबड्स या कोई गिफ्ट।",
}
STARTER_OPTIONS = {
    "en": ["Running shoes", "Laptop", "Earbuds", "Trekking gear", "Pressure cooker", "Gift for father"],
    "hinglish": ["Running shoes", "Laptop", "Earbuds", "Trekking gear", "Pressure cooker", "Papa ke liye gift"],
    "hi": ["जूते", "लैपटॉप", "ईयरबड्स", "ट्रेकिंग बैकपैक", "प्रेशर कुकर", "पापा के लिए गिफ्ट"],
}


def quick_replies(slot: str, matches: list[Product], language: str) -> list[str]:
    """One-tap answers for the question we just asked, built from the real matches (UI buttons)."""
    out: list[str] = []
    if slot == "budget_max":
        prices = sorted(p.price_inr for p in matches if p.price_inr is not None)
        if prices:
            cuts = sorted({_round_up(prices[len(prices) // 3]), _round_up(prices[(2 * len(prices)) // 3])})
            tpl = {"hi": "{} तक", "hinglish": "{} tak"}.get(language, "Under {}")
            out = [tpl.format(format_inr(c)) for c in cuts]
    elif slot == "use":
        counts = Counter(t.lower().strip() for p in matches for t in (p.tags or []) if t.lower().strip() in _USE_TAGS)
        out = [t.capitalize() for t, _ in counts.most_common(3)]
    elif slot == "size":
        out = [f"Size {s}" for s in sorted({s for p in matches for s in (p.sizes or [])}, key=lambda x: (len(x), x))[:4]]
    elif slot == "brand":
        out = [b for b, _ in Counter(p.brand for p in matches if p.brand).most_common(3)]
    return out + [_SHOW_ALL.get(language, "Show all")] if out else []


def next_question(state: SessionState, matches: list[Product], language: str) -> Optional[tuple[str, str]]:
    """`matches` = products that matched the request after filters. Returns (slot, question) or None."""
    f = state.filters
    # one question per product: after the shopper answered once, show results (a second question annoys)
    if state.clarify_streak >= MAX_STREAK or not matches or state.asked_slots:
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
