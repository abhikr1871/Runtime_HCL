"""Step 4: decide whether to ask a clarifying question (max 2 in a row). Owner: M1.

v0: asks for a budget when the request is vague (1-2 content words, no budget) AND either many
products match or the category varies a lot in price/use (laptops, phones...).
TODO(M1): per-category required details, and pick the missing detail whose values vary most
among the matches (highest entropy), e.g. laptops -> budget vs use, shoes -> size.
"""
from __future__ import annotations

import re
from typing import Optional

from app.schemas import Product, SessionState

MAX_STREAK = 2
MIN_MATCHES_TO_ASK = 8
HIGH_VARIANCE_CATEGORIES = {"laptops", "smartphones", "tablets", "televisions"}
_STOP = {"i", "need", "want", "a", "an", "the", "some", "me", "show", "for", "please", "mujhe", "chahiye", "ek", "koi"}

_QUESTIONS = {
    "budget_max": {
        "en": "What's your budget? For example, under ₹2,000 or under ₹5,000.",
        "hinglish": "Aapka budget kitna hai? Jaise ₹2,000 se kam ya ₹5,000 se kam.",
        "hi": "आपका बजट कितना है? जैसे ₹2,000 से कम या ₹5,000 से कम।",
    },
}


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[\wऀ-ॿ]+", text.lower()) if w not in _STOP]


def next_question(state: SessionState, matches: list[Product], language: str) -> Optional[tuple[str, str]]:
    """`matches` = products that matched the request after filters. Returns (slot, question) or None."""
    if state.clarify_streak >= MAX_STREAK or not matches:
        return None
    f = state.filters
    vague = len(_content_words(f.get("query_text", ""))) <= 2
    broad = len(matches) > MIN_MATCHES_TO_ASK or any(p.category in HIGH_VARIANCE_CATEGORIES for p in matches)
    if vague and broad and "budget_max" not in f and "budget_max" not in state.asked_slots:
        return "budget_max", _QUESTIONS["budget_max"].get(language, _QUESTIONS["budget_max"]["en"])
    return None
