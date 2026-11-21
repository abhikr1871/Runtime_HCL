"""Step 4: decide whether to ask ONE clarifying question (max 2 in a row). Owner: M1.

Ask only when the request is genuinely broad, e.g. "I need a laptop", "show me a phone", "need earbuds":
  - exactly one product word and nothing else specific, AND
  - either it is a broad item (laptop, phone, shoes, watch, gift...) or many products match it, AND
  - the shopper has not given any constraint yet (budget, brand, size, colour), AND
  - we have not already asked about that detail in this conversation.

Anything more specific ("wireless earbuds", "running shoes not Nike", "kurta under 1500") is searched directly:
a useless question costs the shopper a turn, and the brief penalises unnecessary follow-ups.

TODO(M1, stretch): choose WHICH detail to ask (budget vs size vs use) by how much it varies among the matches.
"""
from __future__ import annotations

import re
from typing import Optional

from app.schemas import Product, SessionState

MAX_STREAK = 2
MIN_MATCHES_TO_ASK = 8
BROAD_ITEMS = {"laptop", "laptops", "phone", "phones", "smartphone", "smartphones", "mobile", "shoes", "shoe",
               "watch", "watches", "gift", "gifts", "tablet", "tablets", "clothes", "dress", "bag", "bags"}
_FILLER = {
    "i", "need", "want", "a", "an", "the", "some", "me", "show", "for", "please", "looking", "buy", "get", "find",
    "suggest", "recommend", "something", "good", "nice", "new", "any", "my", "to", "is", "can", "you",
    "mujhe", "chahiye", "ek", "koi", "dikhao", "do", "kuch", "chaiye", "chahie",
}
_CONSTRAINTS = ("budget_max", "budget_min", "brands_include", "brands_exclude", "size", "color")

_QUESTIONS = {
    "budget_max": {
        "en": "What's your budget? For example, under ₹2,000 or under ₹5,000.",
        "hinglish": "Aapka budget kitna hai? Jaise ₹2,000 se kam ya ₹5,000 se kam.",
        "hi": "आपका बजट कितना है? जैसे ₹2,000 से कम या ₹5,000 से कम।",
    },
}


def product_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[\wऀ-ॿ]+", text.lower()) if w not in _FILLER and not w.isdigit()]


def next_question(state: SessionState, matches: list[Product], language: str) -> Optional[tuple[str, str]]:
    """`matches` = products that matched the request after filters. Returns (slot, question) or None."""
    f = state.filters
    if state.clarify_streak >= MAX_STREAK or not matches:
        return None
    if any(f.get(k) for k in _CONSTRAINTS):
        return None
    words = product_words(f.get("query_text", ""))
    if len(words) != 1:
        return None
    broad = words[0] in BROAD_ITEMS or len(matches) > MIN_MATCHES_TO_ASK
    if broad and "budget_max" not in state.asked_slots:
        return "budget_max", _QUESTIONS["budget_max"].get(language, _QUESTIONS["budget_max"]["en"])
    return None
