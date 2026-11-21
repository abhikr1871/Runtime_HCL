"""Step 6: no matches -> drop one constraint at a time and say which one blocked results. Owner: M1.

Order: colour, size, brand, then budget raised to the smallest price yielding at least 1 match.
"""
from __future__ import annotations

from typing import Any, Optional

from app.pipeline import filters, retrieval
from app.pipeline.reasons import format_inr
from app.schemas import Product

_ORDER = ["color", "size", "brands_include", "budget_max"]
_LABEL = {
    "color": {"en": "colour", "hinglish": "colour", "hi": "रंग"},
    "size": {"en": "size", "hinglish": "size", "hi": "साइज़"},
    "brands_include": {"en": "brand", "hinglish": "brand", "hi": "ब्रांड"},
    "budget_max": {"en": "budget", "hinglish": "budget", "hi": "बजट"},
}


def relax(f: dict[str, Any], query: str, language: str = "en") -> Optional[tuple[str, list[tuple[Product, float]], str, list[int]]]:
    """Returns (blocking_constraint_label, ranked_nearest, counterfactual_hint_message, extra_allowed_amounts) or None."""
    for key in _ORDER:
        if not f.get(key):
            continue

        loosened = dict(f)
        if key == "budget_max":
            user_budget = f["budget_max"]
            loosened.pop("budget_max")
            candidates = filters.apply(loosened)
            over_budget = [p for p in candidates if p.price_inr is not None and p.price_inr > user_budget]
            if not over_budget:
                continue

            min_price = min(p.price_inr for p in over_budget)
            diff = min_price - user_budget
            at_min_price = [p for p in over_budget if p.price_inr <= min_price]
            count = len(at_min_price)
            ranked = retrieval.rank(query, over_budget, loosened.get("sort"))

            opt_str = "option" if count == 1 else "options"
            if language == "hi":
                msg = f"₹{diff:,} अधिक ({format_inr(min_price)}) में {count} विकल्प उपलब्ध हैं।"
            elif language == "hinglish":
                msg = f"₹{diff:,} zyada ({format_inr(min_price)}) mein {count} {opt_str} milenge."
            else:
                msg = f"₹{diff:,} more ({format_inr(min_price)}) gives {count} {opt_str}."

            label = _LABEL["budget_max"].get(language, "budget")
            extra_allowed = [diff, min_price, user_budget]
            return label, ranked, msg, extra_allowed

        else:
            loosened.pop(key)
            candidates = filters.apply(loosened)
            ranked = retrieval.rank(query, candidates, loosened.get("sort"))
            if ranked:
                count = len(ranked)
                opt_str = "option" if count == 1 else "options"
                label = _LABEL[key].get(language, _LABEL[key]["en"])
                if language == "hi":
                    msg = f"{label} फ़िल्टर हटाने पर {count} विकल्प उपलब्ध हैं।"
                elif language == "hinglish":
                    msg = f"Aapka {label} filter hatane se {count} {opt_str} milenge."
                else:
                    msg = f"Removing your {label} filter gives {count} {opt_str}."

                return label, ranked, msg, []

    return None
