"""Step 4: decide whether to ask a clarifying question (max 2 in a row). Owner: M1.

Asks for a required detail when the request is vague/broad, choosing the detail whose
values vary most among the current candidate matches.
"""
from __future__ import annotations

import re
from typing import Optional

from app.pipeline.reasons import format_inr
from app.schemas import Product, SessionState

MAX_STREAK = 2
MIN_MATCHES_TO_ASK = 3
_STOP = {
    "i", "need", "want", "a", "an", "the", "some", "me", "show", "for", "please",
    "mujhe", "chahiye", "ek", "koi", "in", "with", "good", "best", "give", "dikhaye", "dikhao",
}

_CATEGORY_ATTRIBUTES = {
    "mens-shoes": ["size", "use", "budget_max", "brand"],
    "shoes": ["size", "use", "budget_max", "brand"],
    "footwear": ["size", "use", "budget_max", "brand"],
    "laptops": ["use", "budget_max", "brand"],
    "laptop": ["use", "budget_max", "brand"],
    "mobile-accessories": ["use", "budget_max", "brand"],
    "mens-shirts": ["size", "budget_max", "brand"],
    "shirts": ["size", "budget_max", "brand"],
    "clothing": ["size", "budget_max", "brand"],
    "bags": ["use", "budget_max", "brand"],
    "sports-accessories": ["use", "budget_max", "brand"],
}
_DEFAULT_ATTRIBUTES = ["budget_max", "use", "size", "brand"]

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


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[\wऀ-ॿ]+", text.lower()) if w not in _STOP]


def _attribute_variance(attr: str, matches: list[Product], language: str = "en") -> tuple[int, str]:
    """Computes distinct value count and 2-4 formatted example options."""
    if attr == "size":
        sizes: list[str] = []
        for p in matches:
            sizes.extend(p.sizes or [])
        distinct = sorted(set(sizes), key=lambda x: (len(x), x))
        if len(distinct) < 2:
            return 0, ""
        options = ", ".join(distinct[:4])
        return len(distinct), options

    elif attr == "use":
        tags: list[str] = []
        generic = {"shoes", "shoe", "laptop", "laptops", "phone", "phones", "shirt", "shirts", "bag", "bags", "gift"}
        for p in matches:
            for t in (p.tags or []):
                t_low = t.lower().strip()
                if t_low not in generic:
                    tags.append(t_low)
        distinct = sorted(set(tags))
        if len(distinct) < 2:
            return 0, ""
        options = ", ".join(distinct[:3])
        return len(distinct), options

    elif attr == "brand":
        brands = [p.brand for p in matches if p.brand]
        distinct = sorted(set(brands))
        if len(distinct) < 2:
            return 0, ""
        options = ", ".join(distinct[:3])
        return len(distinct), options

    elif attr == "budget_max":
        prices = [p.price_inr for p in matches if p.price_inr is not None]
        if not prices or len(set(prices)) < 2:
            return 0, ""
        min_p, max_p = min(prices), max(prices)
        if max_p - min_p < 500:
            return 0, ""
        opt1 = format_inr(int(round(min_p * 1.5, -2)))
        opt2 = format_inr(int(round(max_p * 0.9, -2)))
        if language == "hi":
            options = f"{opt1} से कम या {opt2} से कम"
        elif language == "hinglish":
            options = f"{opt1} se kam ya {opt2} se kam"
        else:
            options = f"under {opt1} or under {opt2}"
        return len(set(prices)), options

    return 0, ""


def next_question(state: SessionState, matches: list[Product], language: str) -> Optional[tuple[str, str]]:
    """`matches` = products that matched the request after filters. Returns (slot, question) or None."""
    if state.clarify_streak >= MAX_STREAK or not matches or len(matches) <= MIN_MATCHES_TO_ASK:
        return None

    f = state.filters
    q_text = f.get("query_text", "")
    content = _content_words(q_text)
    # Only clarify when request is vague/broad
    if len(content) > 2:
        return None

    cat = f.get("category")
    if not cat and matches:
        cats = {p.category for p in matches}
        if len(cats) == 1:
            cat = next(iter(cats))

    candidate_attrs = _CATEGORY_ATTRIBUTES.get(cat, _DEFAULT_ATTRIBUTES) if cat else _DEFAULT_ATTRIBUTES

    best_attr, best_var, best_opts = None, 0, ""
    for attr in candidate_attrs:
        if attr in f or attr in state.asked_slots:
            continue
        if attr == "size" and ("size" in f or "size" in state.asked_slots):
            continue

        var, opts = _attribute_variance(attr, matches, language)
        if var > best_var:
            best_var = var
            best_attr = attr
            best_opts = opts

    if best_attr and best_var >= 2:
        template_str = _QUESTIONS.get(best_attr, {}).get(language, _QUESTIONS.get(best_attr, {})["en"])
        question = template_str.format(options=best_opts)
        return best_attr, question

    return None
