"""Normalise messy catalogue values (units, prices). Owner: M2.

The brief plants mixed units (g vs kg) and missing fields on purpose.
Rule: if a value cannot be parsed, return None ("not listed") - never guess.
"""
from __future__ import annotations

import re
from typing import Optional

_WEIGHT = re.compile(r"^\s*([\d.]+)\s*(kg|kgs|kilo|kilogram|kilograms|g|gm|gms|gram|grams|mg)?\s*$", re.I)
_TO_G = {"kg": 1000, "kgs": 1000, "kilo": 1000, "kilogram": 1000, "kilograms": 1000,
         "g": 1, "gm": 1, "gms": 1, "gram": 1, "grams": 1, "mg": 0.001}


def weight_to_grams(raw: Optional[str | float | int]) -> Optional[float]:
    """'1.2 kg' -> 1200.0, '500g' -> 500.0, '4' (no unit) -> None (ambiguous: flag it, don't guess)."""
    if raw is None:
        return None
    m = _WEIGHT.match(str(raw))
    if not m or not m.group(2):
        return None
    return round(float(m.group(1)) * _TO_G[m.group(2).lower()], 3)


def parse_price_inr(raw: Optional[str | float | int]) -> Optional[int]:
    """'Rs. 2,499' / '₹2499' / 2499.0 -> 2499. Unparseable -> None."""
    if raw is None:
        return None
    digits = re.sub(r"[^\d.]", "", str(raw)).strip(".")
    try:
        return int(round(float(digits))) if digits else None
    except ValueError:
        return None

# TODO(M2): sizes (UK/US/EU, cm/inch), colour synonyms, and a fix report for the data note.
