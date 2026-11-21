"""Step 3: merge this turn into the session (follow-ups, negation, references). Owner: M3.

Handles: "cheaper ones", "only Nike", "not Nike", "compare the first two", "is the second one in stock?".
"""
from __future__ import annotations

from app.schemas import Product, SessionState, Slots

_FILTER_KEYS = ["category", "product_type", "budget_min", "budget_max", "size", "color", "occasion", "recipient"]


def apply(state: SessionState, slots: Slots, last_products: list[Product]) -> tuple[SessionState, list[str]]:
    """Returns the updated state and any product IDs the shopper pointed at."""
    f = dict(state.filters)
    follow_up = slots.intent in {"refine", "compare", "product_fact"} or bool(slots.references)

    if not follow_up:
        # TODO(M3): detect a brand-new topic (different product type) and reset old filters
        f["query_text"] = slots.query_text
    for key in _FILTER_KEYS:
        value = getattr(slots, key)
        if value not in (None, "", []):
            f[key] = value

    excluded = set(f.get("brands_exclude", [])) | set(slots.brands_exclude)
    included = (set(slots.brands_include) or set(f.get("brands_include", []))) - excluded
    f["brands_exclude"], f["brands_include"] = sorted(excluded), sorted(included)

    if slots.wants_cheaper:
        prices = [p.price_inr for p in last_products if p.price_inr is not None]
        if prices:
            f["budget_max"] = min(prices) - 1
        f["sort"] = "price_asc"
    elif slots.sort:
        f["sort"] = slots.sort

    selected = [state.last_results[i] for i in slots.references if i < len(state.last_results)]
    selected += [pid for pid in slots.product_ids if pid not in selected]

    new_state = state.model_copy(update={"filters": {k: v for k, v in f.items() if v not in (None, "", [])},
                                         "turn_count": state.turn_count + 1})
    return new_state, selected
