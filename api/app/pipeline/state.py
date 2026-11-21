"""Step 3: merge this turn into the session (follow-ups, negation, references). Owner: M3.

Handles: "cheaper ones", "only Nike", "not Nike", "compare the first two", "is the second one in stock?".
"""
from __future__ import annotations

from app.pipeline import retrieval
from app.schemas import Product, SessionState, Slots

# Words that answer "what will you use it for?" without naming a new product. Anything else after our
# question ("mujhe chappal chahiye" after "laptop budget?") is a NEW product and starts a fresh search.
_ANSWER_WORDS = {"running", "trekking", "hiking", "walking", "jogging", "gym", "fitness", "sports", "yoga", "cricket",
                 "football", "office", "work", "formal", "party", "casual", "wedding", "festive", "travel", "daily",
                 "outdoor", "study", "school", "college", "gaming", "music", "kids", "gift", "coding", "programming",
                 "students", "student", "men", "women", "mens", "womens", "boys", "girls", "father", "mother", "wife",
                 "husband", "friend", "cheap", "budget", "premium", "light", "lightweight", "waterproof", "size",
                 "black", "white", "red", "blue", "green", "grey", "small", "medium", "large", "big"} | {"all", "sab", "any", "koi", "dikhao"}
_NEUTRAL = {"all", "sab", "any", "koi", "dikhao", "budget", "size"}   # "Show all" answers the question, adds no topic
_FILTER_KEYS = ["category", "product_type", "budget_min", "budget_max", "size", "color", "occasion", "recipient"]


def apply(state: SessionState, slots: Slots, last_products: list[Product]) -> tuple[SessionState, list[str]]:
    """Returns the updated state and any product IDs the shopper pointed at."""
    f = dict(state.filters)
    follow_up = slots.intent in {"refine", "compare", "product_fact"} or bool(slots.references)

    new_topic = False
    if not follow_up:
        new_words = retrieval.tokens(slots.query_text or "")
        old_query = f.get("query_text", "")
        old_words = set(retrieval.tokens(old_query)) if old_query else set()
        answers_question = (state.clarify_streak > 0 and old_query
                            and (not new_words or set(new_words) & old_words or set(new_words) <= _ANSWER_WORDS))
        if answers_question:
            # answering our clarifying question ("running", "for office", "under 3000"): add to the topic
            if new_words and not set(new_words) <= old_words:
                extra = [w for w in new_words if w not in old_words and w not in _NEUTRAL]
                if extra:
                    f["query_text"] = f"{old_query} {' '.join(extra)}"
        elif old_query and not new_words:
            pass   # "under 3000", "3000 tak": only a constraint, keep talking about the same product
        else:
            if old_query and not set(new_words) & set(retrieval.tokens(old_query)):
                # a different product ("shoes under 3000" -> "pressure cooker"): start fresh, old budget/brand gone
                f, new_topic = {}, True
            f["query_text"] = slots.query_text
    for key in _FILTER_KEYS:
        value = getattr(slots, key)
        if value not in (None, "", []):
            f[key] = value

    excluded = set(f.get("brands_exclude", [])) | set(slots.brands_exclude)
    included = (set(slots.brands_include) or set(f.get("brands_include", []))) - excluded
    f["brands_exclude"], f["brands_include"] = sorted(excluded), sorted(included)

    if slots.wants_more:
        f["exclude_ids"] = sorted(set(f.get("exclude_ids", [])) | set(state.last_results))
    elif not follow_up:
        f.pop("exclude_ids", None)        # a fresh request may show earlier products again
    if slots.wants_cheaper:
        prices = [p.price_inr for p in last_products if p.price_inr is not None]
        if prices:
            f["budget_max"] = min(prices) - 1
        f["sort"] = "price_asc"
    elif slots.sort:
        f["sort"] = slots.sort

    selected = [state.last_results[i] for i in slots.references if i < len(state.last_results)]
    selected += [pid for pid in slots.product_ids if pid not in selected]

    update = {"filters": {k: v for k, v in f.items() if v not in (None, "", [])}, "turn_count": state.turn_count + 1}
    if new_topic:
        update.update({"asked_slots": [], "clarify_streak": 0})   # a new product may need its own question
    new_state = state.model_copy(update=update)
    return new_state, selected
