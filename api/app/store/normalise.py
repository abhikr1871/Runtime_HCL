"""Normalise messy catalogue values (units, prices, sizes, colours). Owner: M2.

The brief plants mixed units (g vs kg) and missing fields on purpose.
Rule: if a value cannot be parsed, return None ("not listed") - never guess.
"""
from __future__ import annotations

import re
from typing import Any, Optional

_WEIGHT = re.compile(r"^\s*([\d.]+)\s*(kg|kgs|kilo|kilogram|kilograms|g|gm|gms|gram|grams|mg)?\s*$", re.I)
_TO_G = {
    "kg": 1000, "kgs": 1000, "kilo": 1000, "kilogram": 1000, "kilograms": 1000,
    "g": 1, "gm": 1, "gms": 1, "gram": 1, "grams": 1, "mg": 0.001,
}

_SIZE_MAP = {
    "extra small": "XS", "xs": "XS",
    "small": "S", "s": "S",
    "medium": "M", "m": "M",
    "large": "L", "l": "L",
    "extra large": "XL", "xl": "XL", "xlarge": "XL",
    "double extra large": "XXL", "xxl": "XXL", "2xl": "XXL",
}

_COLOR_SYNONYMS = {
    "navy": "blue", "navy blue": "blue", "dark blue": "blue", "sky blue": "blue", "cyan": "blue",
    "maroon": "red", "crimson": "red", "ruby": "red", "burgundy": "red",
    "charcoal": "grey", "gray": "grey", "silver": "grey",
    "golden": "gold",
    "olive": "green", "lime": "green", "emerald": "green",
    "beige": "brown", "tan": "brown", "khaki": "brown",
    "violet": "purple", "lavender": "purple",
}


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


def normalise_size(raw: Optional[str | int]) -> Optional[str]:
    """Standardize shoe/apparel sizes: 'UK 8' -> '8', 'medium' -> 'M', etc."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    lower = s.lower()
    if lower in _SIZE_MAP:
        return _SIZE_MAP[lower]
    # Handle UK/US/EU prefixes
    m = re.match(r"^(?:uk|us|eu|size)?\s*[-:]?\s*(\d{1,2}(?:\.\d)?|[a-z]+)$", lower)
    if m:
        val = m.group(1).upper()
        return _SIZE_MAP.get(val.lower(), val)
    return s.upper()


def normalise_color(raw: Optional[str]) -> Optional[str]:
    """Map color variants and synonyms to standard base colors."""
    if not raw or not isinstance(raw, str):
        return None
    c = raw.strip().lower()
    return _COLOR_SYNONYMS.get(c, c)


def normalise_product(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalises raw product fields and records each fix applied for data note reporting."""
    p = dict(data)
    fixes: list[str] = []

    # 1. Price
    if "price_inr" not in p or p["price_inr"] is None:
        parsed_price = parse_price_inr(p.get("price"))
        if parsed_price is not None:
            p["price_inr"] = parsed_price
            fixes.append(f"parsed price '{p.get('price')}' -> {parsed_price}")

    # 2. Weight
    if ("weight_g" not in p or p["weight_g"] is None) and p.get("weight_raw"):
        wg = weight_to_grams(p["weight_raw"])
        if wg is not None:
            p["weight_g"] = wg
            fixes.append(f"normalised weight '{p['weight_raw']}' -> {wg}g")

    # 3. Sizes
    if "sizes" in p and isinstance(p["sizes"], list):
        norm_sizes = []
        for sz in p["sizes"]:
            nsz = normalise_size(sz)
            if nsz:
                norm_sizes.append(nsz)
        if norm_sizes != p["sizes"]:
            fixes.append(f"standardized sizes {p['sizes']} -> {norm_sizes}")
            p["sizes"] = norm_sizes

    # 4. Colors
    if "colors" in p and isinstance(p["colors"], list):
        norm_colors = []
        for col in p["colors"]:
            ncol = normalise_color(col)
            if ncol and ncol not in norm_colors:
                norm_colors.append(ncol)
        if norm_colors != p["colors"]:
            fixes.append(f"standardized colors {p['colors']} -> {norm_colors}")
            p["colors"] = norm_colors

    return p, fixes
