"""Normalise messy catalogue values (units, prices, brands, sizes, colours). Owner: M2-B.

The brief plants mixed units (g vs kg), missing fields, inconsistent casings and synonyms on purpose.
Rules:
- If a value cannot be parsed, return None ("not listed") - never guess.
- Preserve original raw values (e.g. weight_raw alongside weight_g).
- Never change actual product prices or product IDs.
- Keep normalization deterministic and repeatable.
"""
from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import Any, Optional

from rapidfuzz import fuzz

# ---------- Weight & Unit Normalisation ----------
_WEIGHT = re.compile(r"^\s*([\d.]+)\s*(kg|kgs|kilo|kilogram|kilograms|g|gm|gms|gram|grams|mg)?\s*$", re.I)
_TO_G = {
    "kg": 1000.0, "kgs": 1000.0, "kilo": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "g": 1.0, "gm": 1.0, "gms": 1.0, "gram": 1.0, "grams": 1.0, "mg": 0.001,
}

_VOLUME = re.compile(r"^\s*([\d.]+)\s*(l|litre|litres|liter|liters|ml|millilitre|millilitres)?\s*$", re.I)
_TO_ML = {
    "l": 1000.0, "litre": 1000.0, "litres": 1000.0, "liter": 1000.0, "liters": 1000.0,
    "ml": 1.0, "millilitre": 1.0, "millilitres": 1.0,
}

_DIMENSION_INCH = re.compile(r"^\s*([\d.]+)\s*(inch|inches|in|\"|'')?\s*$", re.I)


def weight_to_grams(raw: Optional[str | float | int]) -> Optional[float]:
    """'1.2 kg' -> 1200.0, '500g' -> 500.0, '4' (no unit) -> None (ambiguous: flag it, don't guess)."""
    if raw is None:
        return None
    m = _WEIGHT.match(str(raw).strip())
    if not m or not m.group(2):
        return None
    try:
        val = float(m.group(1))
        unit = m.group(2).lower()
        return round(val * _TO_G[unit], 3)
    except (ValueError, KeyError):
        return None


def volume_to_ml(raw: Optional[str | float | int]) -> Optional[float]:
    """'1 L' -> 1000.0, '750 ml' -> 750.0, unitless -> None."""
    if raw is None:
        return None
    m = _VOLUME.match(str(raw).strip())
    if not m or not m.group(2):
        return None
    try:
        val = float(m.group(1))
        unit = m.group(2).lower()
        return round(val * _TO_ML[unit], 3)
    except (ValueError, KeyError):
        return None


def dimension_to_inches(raw: Optional[str | float | int]) -> Optional[float]:
    """'15.6 inch' / '14"' -> 15.6 / 14.0. Unitless -> None."""
    if raw is None:
        return None
    m = _DIMENSION_INCH.match(str(raw).strip())
    if not m or not m.group(2):
        return None
    try:
        return round(float(m.group(1)), 2)
    except ValueError:
        return None


# ---------- Price Normalisation ----------
def parse_price_inr(raw: Optional[str | float | int]) -> Optional[int]:
    """'Rs. 2,499' / '₹2499' / 2499.0 -> 2499. Negative/0 or unparseable -> None."""
    if raw is None:
        return None
    s = str(raw).strip()
    if "-" in s:
        return None
    digits = re.sub(r"[^\d.]", "", s).strip(".")
    try:
        val = int(round(float(digits))) if digits else None
        return val if (val is not None and val > 0) else None
    except ValueError:
        return None


# ---------- Brand Normalisation ----------
# Canonical brand lookup table covering common Indian & global tech/apparel brands
_KNOWN_BRANDS: dict[str, str] = {
    "boat": "boAt",
    "wildcraft": "Wildcraft",
    "quechua": "Quechua",
    "decathlon": "Decathlon",
    "milton": "Milton",
    "lenovo": "Lenovo",
    "asus": "ASUS",
    "nike": "Nike",
    "campus": "Campus",
    "fabindia": "Fabindia",
    "apple": "Apple",
    "samsung": "Samsung",
    "prestige": "Prestige",
    "hawkins": "Hawkins",
    "pigeon": "Pigeon",
    "bajaj": "Bajaj",
    "philips": "Philips",
    "wonderchef": "Wonderchef",
    "fastrack": "Fastrack",
    "titan": "Titan",
    "realme": "Realme",
    "oneplus": "OnePlus",
    "xiaomi": "Xiaomi",
    "mi": "Xiaomi",
    "hp": "HP",
    "hewlett-packard": "HP",
    "dell": "Dell",
    "acer": "Acer",
    "puma": "Puma",
    "adidas": "Adidas",
    "reebok": "Reebok",
    "woodland": "Woodland",
    "peter england": "Peter England",
    "allen solly": "Allen Solly",
    "van heusen": "Van Heusen",
    "raymond": "Raymond",
}


def normalize_brand(raw: Optional[str]) -> Optional[str]:
    """Returns canonical brand casing if recognized; otherwise cleans extra spaces.
    None or empty -> None (never guess unlisted brands).
    """
    if raw is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(raw)).strip()
    if not cleaned:
        return None
    lower = cleaned.lower()
    if lower in _KNOWN_BRANDS:
        return _KNOWN_BRANDS[lower]
    # Preserve reasonable title casing if input was all lowercase or all uppercase
    if cleaned.islower() or cleaned.isupper():
        return cleaned.title()
    return cleaned


# ---------- Category Normalisation ----------
_CATEGORY_MAP: dict[str, str] = {
    "mens-shoes": "mens-shoes",
    "men-shoes": "mens-shoes",
    "mens shoes": "mens-shoes",
    "men's shoes": "mens-shoes",
    "womens-shoes": "womens-shoes",
    "women-shoes": "womens-shoes",
    "womens shoes": "womens-shoes",
    "women's shoes": "womens-shoes",
    "shoes": "shoes",
    "laptops": "laptops",
    "laptop": "laptops",
    "bags": "bags",
    "bag": "bags",
    "backpacks": "bags",
    "backpack": "bags",
    "sports-accessories": "sports-accessories",
    "sports accessories": "sports-accessories",
    "mobile-accessories": "mobile-accessories",
    "mobile accessories": "mobile-accessories",
    "mens-shirts": "mens-shirts",
    "men-shirts": "mens-shirts",
    "mens shirts": "mens-shirts",
    "men's shirts": "mens-shirts",
    "womens-dresses": "womens-dresses",
    "women-dresses": "womens-dresses",
    "womens dresses": "womens-dresses",
    "women's dresses": "womens-dresses",
    "kitchen": "kitchen",
    "kitchen-accessories": "kitchen",
    "kitchen accessories": "kitchen",
    "kitchen appliances": "kitchen",
    "fragrances": "fragrances",
    "groceries": "groceries",
    "beauty": "beauty",
    "home-decoration": "home-decoration",
    "home decoration": "home-decoration",
    "smartphones": "smartphones",
    "smartphone": "smartphones",
}


def normalize_category(raw: Optional[str]) -> Optional[str]:
    """Standardizes category strings to consistent slug/canonical names."""
    if raw is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(raw)).strip().lower()
    if not cleaned:
        return None
    return _CATEGORY_MAP.get(cleaned, cleaned.replace(" ", "-"))


# ---------- Size Normalisation ----------
_CLOTHING_SIZES: dict[str, str] = {
    "xs": "XS", "extra small": "XS",
    "s": "S", "small": "S",
    "m": "M", "medium": "M",
    "l": "L", "large": "L",
    "xl": "XL", "extra large": "XL",
    "xxl": "XXL", "2xl": "XXL", "double extra large": "XXL",
    "xxxl": "3XL", "3xl": "3XL",
    "4xl": "4XL", "xxxxl": "4XL",
}

_SHOE_SIZE_RE = re.compile(r"^(?:uk|us|eu|size|ind)?[\s\-_]*([\d.]+)(?:[\s\-_]*(?:uk|us|eu))?$", re.I)


def normalize_size(raw: Optional[str | int | float]) -> Optional[str]:
    """Normalizes shoe sizes ('UK 8' -> '8'), clothing sizes ('large' -> 'L'), or dimensions."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    lower = s.lower()
    if lower in _CLOTHING_SIZES:
        return _CLOTHING_SIZES[lower]
    m = _SHOE_SIZE_RE.match(s)
    if m:
        num = m.group(1)
        # Format whole numbers as integer string (e.g. 8.0 -> 8)
        try:
            f = float(num)
            return str(int(f)) if f.is_integer() else str(f)
        except ValueError:
            return num
    # Check for inch pattern like '14 inch' / '14"'
    m_inch = re.match(r"^([\d.]+)\s*(?:inch|inches|in|\"|'')$", s, re.I)
    if m_inch:
        return f"{m_inch.group(1)} inch"
    return s


def normalize_sizes(raw_list: Optional[list[Any]]) -> list[str]:
    """Normalizes a list of sizes, deduplicating while preserving order."""
    if not raw_list:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw_list:
        norm = normalize_size(item)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# ---------- Colour Normalisation ----------
# Multilingual and synonym colour mapping (English, Hindi, Hinglish)
_COLOR_MAP: dict[str, str] = {
    # Blue
    "blue": "blue", "navy": "blue", "navy blue": "blue", "royal blue": "blue",
    "sky blue": "blue", "dark blue": "blue", "light blue": "blue", "नीला": "blue", "neela": "blue",
    # Black
    "black": "black", "dark black": "black", "jet black": "black", "matte black": "black",
    "काला": "black", "kaala": "black",
    # Red
    "red": "red", "bright red": "red", "crimson": "red", "maroon": "red", "scarlet": "red",
    "लाल": "red", "laal": "red",
    # Green
    "green": "green", "dark green": "green", "olive": "green", "olive green": "green",
    "light green": "green", "forest green": "green", "हरा": "green", "hara": "green",
    # White
    "white": "white", "off white": "white", "off-white": "white", "pure white": "white",
    "ivory": "white", "सफ़ेद": "white", "safed": "white",
    # Grey
    "grey": "grey", "gray": "grey", "charcoal": "grey", "slate": "grey",
    "स्लेटी": "grey", "graey": "grey",
    # Yellow
    "yellow": "yellow", "lemon": "yellow", "mustard": "yellow", "पीला": "yellow", "peela": "yellow",
    # Orange
    "orange": "orange", "tangerine": "orange", "नारंगी": "orange", "narangi": "orange",
    # Brown
    "brown": "brown", "tan": "brown", "chocolate": "brown", "beige": "brown",
    "भूरा": "brown", "bhoora": "brown",
    # Pink
    "pink": "pink", "rose": "pink", "magenta": "pink", "गुलाबी": "pink", "gulabi": "pink",
    # Purple
    "purple": "purple", "violet": "purple", "lavender": "purple", "बैंगनी": "purple",
    # Gold & Silver
    "gold": "gold", "golden": "gold", "सुनहरा": "gold",
    "silver": "silver", "metallic": "silver", "चांदी": "silver",
}


def normalize_color(raw: Optional[str]) -> Optional[str]:
    """Maps color synonyms, shades, and Hindi/Hinglish terms to base canonical color."""
    if raw is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(raw)).strip().lower()
    if not cleaned:
        return None
    return _COLOR_MAP.get(cleaned, cleaned)


def normalize_colors(raw_list: Optional[list[Any]]) -> list[str]:
    """Normalizes a list of colors, deduplicating while preserving order."""
    if not raw_list:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw_list:
        norm = normalize_color(item)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# ---------- Full Product Normalisation & Audit ----------
def normalize_product(raw: dict[str, Any], record_fixes: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalizes product data dictionary safely:
    - Normalizes brand, category, sizes, colors, weight_g, price_inr, stock.
    - Preserves weight_raw and raw attributes.
    - Never invents missing attributes (missing brand remains None).
    - Returns (normalized_dict, fixes_list).
    """
    p = copy.deepcopy(raw)
    fixes: list[dict[str, Any]] = []
    pid = str(p.get("product_id") or "UNKNOWN")

    def fix(field: str, before: Any, after: Any, reason: str) -> None:
        if record_fixes and before != after:
            fixes.append({"product_id": pid, "field": field, "before": before, "after": after, "reason": reason})

    # 1. Brand
    orig_brand = p.get("brand")
    norm_brand = normalize_brand(orig_brand)
    if orig_brand != norm_brand:
        fix("brand", orig_brand, norm_brand, "canonical brand casing")
        p["brand"] = norm_brand

    # 2. Category
    orig_cat = p.get("category")
    norm_cat = normalize_category(orig_cat)
    if norm_cat and orig_cat != norm_cat:
        fix("category", orig_cat, norm_cat, "canonical category slug")
        p["category"] = norm_cat

    # 3. Sizes
    orig_sizes = p.get("sizes") or []
    if isinstance(orig_sizes, str):
        orig_sizes = [s.strip() for s in orig_sizes.split(",") if s.strip()]
    norm_sizes = normalize_sizes(orig_sizes)
    if orig_sizes != norm_sizes:
        fix("sizes", orig_sizes, norm_sizes, "standardized size format")
        p["sizes"] = norm_sizes
    else:
        p["sizes"] = norm_sizes

    # 4. Colors
    orig_colors = p.get("colors") or []
    if isinstance(orig_colors, str):
        orig_colors = [c.strip() for c in orig_colors.split(",") if c.strip()]
    norm_colors = normalize_colors(orig_colors)
    if orig_colors != norm_colors:
        fix("colors", orig_colors, norm_colors, "canonical color synonym")
        p["colors"] = norm_colors
    else:
        p["colors"] = norm_colors

    # 5. Weight
    orig_w_raw = p.get("weight_raw")
    orig_w_g = p.get("weight_g")
    if orig_w_g is None and orig_w_raw:
        parsed_g = weight_to_grams(orig_w_raw)
        if parsed_g is not None:
            fix("weight_g", None, parsed_g, f"converted {orig_w_raw} to grams")
            p["weight_g"] = parsed_g
        else:
            fix("weight_g", None, None, f"ambiguous unit in {orig_w_raw}: kept as not listed")

    # 6. Price
    orig_price = p.get("price_inr")
    if orig_price is not None:
        parsed_price = parse_price_inr(orig_price)
        if parsed_price != orig_price:
            fix("price_inr", orig_price, parsed_price, "parsed price integer")
            p["price_inr"] = parsed_price

    # 7. MRP
    orig_mrp = p.get("mrp_inr")
    if orig_mrp is not None:
        parsed_mrp = parse_price_inr(orig_mrp)
        if parsed_mrp != orig_mrp:
            fix("mrp_inr", orig_mrp, parsed_mrp, "parsed mrp integer")
            p["mrp_inr"] = parsed_mrp

    # 8. Stock
    stock = p.get("stock", 0)
    try:
        stock_int = max(0, int(stock))
    except (ValueError, TypeError):
        stock_int = 0
    if stock != stock_int:
        fix("stock", stock, stock_int, "non-negative integer stock")
        p["stock"] = stock_int

    # 9. Tags
    tags = p.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip().lower() for t in tags.split(",") if t.strip()]
    p["tags"] = [str(t).strip().lower() for t in tags if str(t).strip()]

    return p, fixes


# ---------- Duplicate & Messy Data Detection ----------
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def detect_duplicates(products: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyzes a product catalogue and produces a report of:
    - Exact duplicate product IDs
    - Identical product titles
    - Near-duplicate product titles (using RapidFuzz)
    - Missing essential fields (brand, description, category)
    - Invalid prices (<= 0 or missing)
    - Out of stock products
    - Multilingual description breakdown (Hindi, Hinglish, English)
    """
    id_counts: dict[str, list[dict]] = defaultdict(list)
    title_groups: dict[str, list[str]] = defaultdict(list)
    invalid_records: list[dict[str, Any]] = []
    out_of_stock_ids: list[str] = []
    missing_brands: list[str] = []
    missing_descriptions: list[str] = []
    lang_counts: dict[str, int] = defaultdict(int)

    for p in products:
        pid = str(p.get("product_id") or "MISSING_ID")
        name = str(p.get("name") or "").strip()
        cat = str(p.get("category") or "").strip()
        price = p.get("price_inr")
        stock = p.get("stock", 0)
        desc = str(p.get("description") or "").strip()
        brand = p.get("brand")

        id_counts[pid].append(p)
        if name:
            clean_title = re.sub(r"\s+", " ", name.lower())
            title_groups[clean_title].append(pid)
        else:
            invalid_records.append({"product_id": pid, "reason": "missing name"})

        if not cat:
            invalid_records.append({"product_id": pid, "reason": "missing category"})

        if price is None or price <= 0:
            invalid_records.append({"product_id": pid, "reason": f"invalid price: {price}"})

        if stock <= 0:
            out_of_stock_ids.append(pid)

        if not brand:
            missing_brands.append(pid)

        if not desc:
            missing_descriptions.append(pid)

        lang = p.get("description_lang", "en")
        if bool(DEVANAGARI.search(desc)):
            lang = "hi"
        lang_counts[lang] += 1

    # Exact duplicate IDs
    exact_duplicate_ids = [pid for pid, items in id_counts.items() if len(items) > 1]

    # Duplicate titles (same title across different IDs)
    duplicate_titles = {title: pids for title, pids in title_groups.items() if len(pids) > 1}

    # Near-duplicate title detection using RapidFuzz
    near_duplicate_groups: list[dict[str, Any]] = []
    processed_titles = list(title_groups.keys())
    for i in range(len(processed_titles)):
        t1 = processed_titles[i]
        matched_titles = [t1]
        for j in range(i + 1, len(processed_titles)):
            t2 = processed_titles[j]
            # Fast token set ratio
            sim = fuzz.token_set_ratio(t1, t2)
            if sim >= 88:
                matched_titles.append(t2)
        if len(matched_titles) > 1:
            all_pids = [pid for t in matched_titles for pid in title_groups[t]]
            if len(all_pids) > 1 and not any(set(all_pids).issubset(set(g["product_ids"])) for g in near_duplicate_groups):
                near_duplicate_groups.append({
                    "titles": matched_titles,
                    "product_ids": all_pids,
                    "sample_title": t1,
                })

    summary = {
        "total_products": len(products),
        "unique_product_ids": len(id_counts),
        "exact_duplicate_id_count": len(exact_duplicate_ids),
        "exact_duplicate_title_groups": len(duplicate_titles),
        "near_duplicate_groups": len(near_duplicate_groups),
        "invalid_records_count": len(invalid_records),
        "out_of_stock_count": len(out_of_stock_ids),
        "missing_brand_count": len(missing_brands),
        "missing_description_count": len(missing_descriptions),
        "language_breakdown": dict(lang_counts),
    }

    return {
        "summary": summary,
        "exact_duplicate_ids": exact_duplicate_ids,
        "duplicate_titles": duplicate_titles,
        "near_duplicate_groups": near_duplicate_groups,
        "invalid_records": invalid_records,
        "out_of_stock_ids": out_of_stock_ids,
        "missing_brands": missing_brands,
        "missing_descriptions": missing_descriptions,
    }
