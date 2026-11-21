"""Step 2a: rules-first understanding (fast, no LLM). Owner: M1.

Extracts language, budget, brands (incl. negation and typo-tolerance), size, colour, references and intent.
The LLM (nlu_llm.py) is only asked when these rules leave the request unclear.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from rapidfuzz import distance, fuzz

from app.schemas import Slots

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_DEV_NUM = str.maketrans("०१२३४५६७८९", "0123456789")

_HINGLISH_WORDS = {
    "mujhe", "chahiye", "sasta", "sasti", "saste", "kitna", "kitne", "hai", "kya", "ka", "ki", "ke", "se",
    "kam", "wala", "wali", "dikhao", "acha", "accha", "nahi", "nahin", "tak", "mein", "aur", "bhi", "liye", "koi",
}

_WORDS_TO_NUM: dict[str, int] = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5,
    "chhe": 6, "che": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
    "gyarah": 11, "barah": 12, "terah": 13, "chaudah": 14, "pandrah": 15,
    "solah": 16, "satrah": 17, "atharah": 18, "unnis": 19, "bees": 20,
    "tees": 30, "chalis": 40, "pachas": 50, "pachaas": 50,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छ": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
}

_MULT: dict[str, int] = {
    "k": 1000,
    "thousand": 1000,
    "hazaar": 1000,
    "hazar": 1000,
    "hajar": 1000,
    "हज़ार": 1000,
    "हजार": 1000,
    "sau": 100,
    "सौ": 100,
    "lakh": 100000,
    "lac": 100000,
    "लाख": 100000,
}

_ROMAN_WORDS = r"\b(?:ek|do|teen|char|chaar|paanch|panch|chhe|che|saat|aath|nau|das|gyarah|barah|terah|chaudah|pandrah|solah|satrah|atharah|unnis|bees|tees|chalis|pachas|pachaas)\b"
_DEV_WORDS = r"(?<![ऀ-ॿ])(?:एक|दो|तीन|चार|पाँच|पांच|छह|छ|सात|आठ|नौ|दस)(?![ऀ-ॿ])"
_WORDS_PAT = f"(?:{_ROMAN_WORDS}|{_DEV_WORDS})"

_MULT_ROMAN = r"(?:k\b|\b(?:thousand|hazaar|hazar|hajar|sau|lakh|lac)\b)"
_MULT_DEV = r"(?<![ऀ-ॿ])(?:हज़ार|हजार|सौ|लाख)(?![ऀ-ॿ])"
_MULT_PAT = f"(?:{_MULT_ROMAN}|{_MULT_DEV})"

_UNIT_GUARD = r"(?!\s*(?:gb|tb|mb|mp|g|inch(?:es)?|hz|khz|mhz|ghz|mah|ram|cores?)\b)"

_NUM = (
    r"(?:(?:rs\.?|inr|₹)\s*)?"
    r"(?:(?<![\d.])(\d[\d,]*(?:\.\d+)?)(?![\d.])|(" + _WORDS_PAT + r"))"
    r"(?:\s*(" + _MULT_PAT + r"))?"
    + _UNIT_GUARD
)


def _parse_val(num_str: Optional[str], word_str: Optional[str], mult_str: Optional[str]) -> Optional[int]:
    if num_str:
        val = float(num_str.replace(",", ""))
    elif word_str:
        val = float(_WORDS_TO_NUM[word_str.lower()])
    else:
        return None
    if mult_str:
        m = _MULT.get(mult_str.lower(), 1)
        val *= m
    return int(val)


_BETWEEN = [
    re.compile(r"between\s*" + _NUM + r"\s*(?:and|to|-)\s*" + _NUM, re.I),
    re.compile(_NUM + r"\s*(?:se|to|-|से)\s*" + _NUM + r"\s*(?:ke\s+(?:beech|bich)|के\s*बीच|tak|तक)", re.I),
]
_MAX = [
    re.compile(r"(?:under|below|less than|within|up\s?to|upto|max(?:imum)?|budget(?:\s+of|\s+is)?)\s*" + _NUM, re.I),
    re.compile(_NUM + r"\s*(?:se\s+kam|से\s*कम|ke\s+andar|के\s*अंदर|ke\s+neeche|tak(?:\s+ka)?|तक(?:\s*का)?|or less|budget)", re.I),
]
_MIN = [
    re.compile(r"(?:above|over|more than|at least|min(?:imum)?|starting(?:\s+from)?|start(?:\s+from)?)\s*" + _NUM, re.I),
    re.compile(_NUM + r"\s*(?:se\s+(?:zyada|jyada|upar|adhik)|से\s*(?:ज़्यादा|ज्यादा|ऊपर|अधिक)|or more)", re.I),
]
_BARE_CURRENCY = re.compile(r"(?:rs\.?|inr|₹)\s*" + _NUM, re.I)

_NEG = [
    re.compile(r"\b(?:not|no|except|without|but not)\s+([a-z][\w&-]+)", re.I),
    re.compile(r"\b([a-z][\w&-]+)\s+(?:nahi|nahin|mat)\b", re.I),
]
_ONLY = re.compile(r"\bonly\s+([a-z][\w&-]+)", re.I)
_SIZE = re.compile(r"\bsize\s*(\d{1,2}|xs|s|m|l|xl|xxl)\b|\b(?:uk|us|eu)\s*(\d{1,2})\b", re.I)
_COLORS = {"black", "white", "blue", "red", "green", "grey", "gray", "pink", "brown", "yellow", "silver", "gold", "navy", "purple"}
_PRODUCT_ID = re.compile(r"\bP-\d{4,}\b", re.I)
_REFS = [
    (re.compile(r"\b(first two|top two|these two|both|pehle do)\b", re.I), [0, 1]),
    (re.compile(r"\b(first|1st|pehla|pehli)\b(?!\s+two)", re.I), [0]),
    (re.compile(r"\b(second|2nd|dusra|dusri)\b", re.I), [1]),
    (re.compile(r"\b(third|3rd|teesra|teesri)\b", re.I), [2]),
]
_COMPARE = re.compile(r"\bcompare\b|\bvs\.?\b|\bversus\b|\bdifference between\b|\bmukabla\b", re.I)
_CHEAPER = re.compile(r"\bcheaper\b|\bless expensive\b|\bsaste\b|\bkam (?:daam|price|keemat)\b", re.I)
_FACT = re.compile(r"\b(in stock|available|how much (?:is|does)|price of|kitne ka|kitni keemat)\b", re.I)
_GREETING = re.compile(r"^\s*(hi+|hello|hey|namaste|namaskar)\b[\s!.]*$", re.I)
_OUT_OF_SCOPE = re.compile(r"\b(weather|cricket score|election|politics|joke|movie|recipe)\b", re.I)

_BRAND_STOPWORDS = {
    "shoes", "shoe", "phone", "phones", "under", "below", "black", "white", "blue", "green", "red",
    "price", "show", "best", "cheap", "cheaper", "gift", "gifts", "good", "laptop", "laptops",
    "watch", "watches", "shirt", "shirts", "sound", "size", "with", "from", "that", "this",
    "free", "deal", "more", "less", "sale", "home", "wear", "mens", "womens", "boys", "girls",
    "chahiye", "dikhao", "sasta", "saste", "accha", "acha", "wala", "wali", "wale", "karo", "karna",
    "only", "nahi", "nahin", "mat", "not", "between", "over", "above", "item", "items", "both",
    "boot", "boots", "camp", "camps", "cut", "jeans", "sandal", "sandals", "slipper", "slippers",
    "sneaker", "sneakers", "heels", "kurta", "saree", "hoodie", "jacket", "bag", "bags",
    "speaker", "speakers", "earbuds", "headphones", "charger", "cable", "mouse", "keyboard",
    "tablet", "camera", "fridge", "level", "sonic", "bottle", "bottles", "backpack", "backpacks",
    "cotton", "sports", "running", "casual", "formal", "dress", "pants", "tshirt", "tshirts",
}


def _match_brand(token: str, brands: dict[str, str], threshold: float = 85.0) -> Optional[str]:
    """Matches a token to known brands: exact match first, then typo-tolerant rapidfuzz."""
    t = token.lower()
    if t in brands:
        return brands[t]
    if len(t) < 4 or t in _BRAND_STOPWORDS:
        return None
    best_brand, best_score = None, 0.0
    for b_lower, b_orig in brands.items():
        if " " in b_lower:
            continue
        # Tightened fuzzy rules
        if t[0] != b_lower[0]:
            continue
        if abs(len(t) - len(b_lower)) > 1:
            continue
        dl = distance.DamerauLevenshtein.distance(t, b_lower)
        if len(t) <= 5 and dl > 1:
            continue
        ratio = fuzz.ratio(t, b_lower)
        jw = distance.JaroWinkler.similarity(t, b_lower) * 100.0
        score = max(ratio, jw) if dl <= 1 else ratio
        if score >= threshold and score > best_score:
            best_score = score
            best_brand = b_orig
    return best_brand


def detect_language(text: str) -> str:
    if _DEVANAGARI.search(text):
        return "hi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    return "hinglish" if len(words & _HINGLISH_WORDS) >= 2 else "en"


def parse(message: str, known_brands: Iterable[str] = ()) -> Slots:
    text = message.strip()
    s = Slots(query_text=text, language=detect_language(text))
    t_budget = text.translate(_DEV_NUM)

    # 1. Range first
    range_found = False
    for p in _BETWEEN:
        if m := p.search(t_budget):
            s.budget_min = _parse_val(m.group(1), m.group(2), m.group(3))
            s.budget_max = _parse_val(m.group(4), m.group(5), m.group(6))
            range_found = True
            break

    # 2. Contextual min and max
    if not range_found:
        for p in _MIN:
            if m := p.search(t_budget):
                s.budget_min = _parse_val(m.group(1), m.group(2), m.group(3))
                break
        for p in _MAX:
            if m := p.search(t_budget):
                s.budget_max = _parse_val(m.group(1), m.group(2), m.group(3))
                break
        # 3. Bare currency fallback only if no between, min or max matched
        if s.budget_min is None and s.budget_max is None:
            if m := _BARE_CURRENCY.search(t_budget):
                s.budget_max = _parse_val(m.group(1), m.group(2), m.group(3))

    # Brands handling: exact, multi-word, typo-tolerant, negation, only
    brands = {b.lower(): b for b in known_brands if b}
    text_lower = text.lower()

    # Multi-word brand matching
    for b_lower, b_orig in brands.items():
        if " " in b_lower and b_lower in text_lower:
            neg_pattern = (
                r"\b(?:not|no|except|without|but not)\s+" + re.escape(b_lower) + r"\b|"
                r"\b" + re.escape(b_lower) + r"\s+(?:nahi|nahin|mat)\b"
            )
            if re.search(neg_pattern, text_lower):
                if b_orig not in s.brands_exclude:
                    s.brands_exclude.append(b_orig)
            else:
                if b_orig not in s.brands_include and b_orig not in s.brands_exclude:
                    s.brands_include.append(b_orig)

    # Single-word brand negation and only (with typo tolerance)
    for p in _NEG:
        for word in p.findall(text):
            if matched := _match_brand(word, brands):
                if matched not in s.brands_exclude:
                    s.brands_exclude.append(matched)

    only_matched = set()
    for word in _ONLY.findall(text):
        if matched := _match_brand(word, brands):
            only_matched.add(matched)

    # Single-word brand inclusion (with typo tolerance)
    tokens = re.findall(r"[a-z][\w&-]+", text_lower)
    for t in tokens:
        matched = _match_brand(t, brands)
        if matched and matched not in s.brands_exclude and matched not in s.brands_include:
            if only_matched and matched not in only_matched:
                continue
            s.brands_include.append(matched)

    if m := _SIZE.search(text):
        s.size = (m.group(1) or m.group(2)).upper()
    s.color = next((t for t in tokens if t in _COLORS), None)
    s.product_ids = [pid.upper() for pid in _PRODUCT_ID.findall(text)]
    for pattern, idx in _REFS:
        if pattern.search(text):
            s.references = idx
            break

    if _GREETING.match(text):
        s.intent = "greeting"
    elif _OUT_OF_SCOPE.search(text):
        s.intent = "out_of_scope"
    elif _COMPARE.search(text):
        s.intent = "compare"
    elif (s.product_ids or s.references) and _FACT.search(text):
        s.intent = "product_fact"
    elif _CHEAPER.search(text):
        s.wants_cheaper, s.intent = True, "refine"
    elif s.references or (s.brands_include or s.brands_exclude) and len(tokens) <= 4:
        s.intent = "refine"

    found = any([s.budget_min, s.budget_max, s.brands_include, s.brands_exclude, s.size, s.color, s.references, s.product_ids])
    s.confidence = 0.8 if found or s.intent != "search" else (0.6 if len(tokens) >= 3 else 0.4)
    return s
