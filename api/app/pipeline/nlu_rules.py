"""Step 2a: rules-first understanding (fast, no LLM). Owner: M1.

Turns the shopper's message into Slots:
- language (English / Hindi in Devanagari / Hinglish)
- Hindi and Hinglish product words -> English search words ("कुर्ता" -> kurta, "joote" -> shoes)
- typo correction against the catalogue's own vocabulary ("blutooth" -> bluetooth, "nkie" -> Nike)
- budget in English, Hinglish and Hindi ("under 3k", "2000 se kam", "2000 रुपये से कम", "3 hazaar tak")
- brands to include / exclude ("only Nike", "not boAt", "Nike nahi", "without Fabindia")
- size, colour, explicit product IDs, references ("first two", "the second one")
- intent: search, refine, compare, product_fact, greeting, out_of_scope, not_sold

The LLM (nlu_llm.py) is only asked when these rules leave the request unclear.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from rapidfuzz.distance import DamerauLevenshtein

from app.schemas import Slots

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_HINGLISH_WORDS = {
    "mujhe", "chahiye", "sasta", "sasti", "saste", "kitna", "kitne", "hai", "kya", "ka", "ki", "ke", "se",
    "kam", "wala", "wali", "dikhao", "acha", "accha", "nahi", "nahin", "tak", "mein", "aur", "bhi", "liye", "koi",
    "mere", "meri", "mera", "papa", "mummy", "andar", "bata", "batao", "dena", "do", "chaiye", "chahie",
}
# Words never "corrected" as typos (function words, Hinglish, shopping verbs)
_KEEP = _HINGLISH_WORDS | {
    "show", "need", "want", "looking", "please", "some", "under", "below", "above", "between", "only", "with",
    "without", "except", "cheap", "cheaper", "best", "good", "gift", "father", "mother", "brother", "sister",
    "friend", "wife", "husband", "compare", "first", "second", "third", "stock", "available", "price", "budget",
    "something", "anything", "which", "what", "that", "this", "these", "those", "your", "have", "does", "sell",
    "like", "likes", "love", "loves", "rupees", "thousand", "hazaar", "hazar",
    # short English filler (never product words)
    "is", "in", "on", "it", "of", "to", "for", "an", "me", "my", "the", "are", "do", "can", "you", "and", "or",
    "one", "ones", "option", "options", "item", "items", "product", "products", "them", "more", "other", "others",
    "any", "also", "just", "who", "will", "would", "get", "buy", "find", "suggest", "recommend", "kuch", "ek",
}

# Hindi (Devanagari) and Hinglish product words -> English catalogue words
_HINDI_TERMS = {
    "प्रेशर कुकर": "pressure cooker", "कुकर": "cooker", "कुर्ता": "kurta", "कुर्ते": "kurta", "कुर्ती": "kurti",
    "ईयरबड्स": "earbuds", "इयरबड्स": "earbuds", "ईयरफ़ोन": "earphones", "ईयरफोन": "earphones",
    "हेडफ़ोन": "headphones", "हेडफोन": "headphones", "जूते": "shoes", "जूता": "shoes", "रनिंग": "running",
    "ट्रेकिंग": "trekking", "बैकपैक": "backpack", "बैग": "bag", "स्पीकर": "speaker", "ब्लूटूथ": "bluetooth",
    "फ़ोन": "phone", "फोन": "phone", "मोबाइल": "phone", "घड़ी": "watch", "घडी": "watch", "लैपटॉप": "laptop",
    "बोतल": "bottle", "तवा": "tawa", "मिक्सर": "mixer", "ग्राइंडर": "grinder", "परफ्यूम": "perfume",
    "इत्र": "perfume", "चश्मा": "sunglasses", "धूप का चश्मा": "sunglasses", "हेडलैंप": "headlamp",
    "टॉर्च": "headlamp", "साड़ी": "saree", "शर्ट": "shirt", "किताब": "book", "क्रीम": "cream",
}
_HINGLISH_TERMS = {
    "joote": "shoes", "joota": "shoes", "jute": "shoes", "juta": "shoes", "ghadi": "watch", "chashma": "sunglasses",
    "botal": "bottle", "mobile": "phone", "kapde": "clothes", "kurte": "kurta",
}
# Extra words added to the SEARCH text only (catalogue uses category plurals such as "smartphones", "watches")
_SEARCH_SYNONYMS = {
    "phone": "smartphones smartphone", "phones": "smartphones smartphone", "mobile": "smartphones",
    "smartphone": "smartphones", "watch": "watches", "laptop": "laptops", "shoe": "shoes", "earbud": "earbuds",
    "bag": "bags backpack", "sunglass": "sunglasses", "perfume": "fragrances", "earphones": "earbuds neckband",
    "headphones": "earbuds", "tablet": "tablets", "cooker": "pressure",
}

_NUM = r"(?:rs\.?|inr|₹)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b|thousand|hazaar|hazar|हज़ार|हजार)?"
_MAX = [re.compile(r"(?:under|below|less than|within|up\s?to|max(?:imum)?|budget(?:\s+of|\s+is)?)\s*" + _NUM, re.I),
        re.compile(_NUM + r"\s*(?:rs\.?|rupees|रुपये|रुपए|रु\.?)?\s*(?:se\s+kam|ke\s+andar|ke\s+neeche|tak|or less|"
                   r"से\s+कम|तक|के\s+अंदर|के\s+नीचे)", re.I)]
_MIN = [re.compile(r"(?:above|over|more than|at least|min(?:imum)?)\s*" + _NUM, re.I)]
_BETWEEN = re.compile(r"between\s*" + _NUM + r"\s*(?:and|to|-)\s*" + _NUM, re.I)
_NEG = [re.compile(r"\b(?:not|no|except|without|but not)\s+([a-z][\w&-]+)", re.I),
        re.compile(r"\b([a-z][\w&-]+)\s+(?:nahi|nahin|mat)\b", re.I)]
_NEG_WORDS = {"not", "no", "except", "without", "but", "nahi", "nahin", "mat", "only", "show", "me", "the", "a"}
_ONLY = re.compile(r"\bonly\s+([a-z][\w&-]+)", re.I)
_SIZE = re.compile(r"\bsize\s*(\d{1,2}|xs|s|m|l|xl|xxl)\b|\b(?:uk|us|eu)\s*(\d{1,2})\b", re.I)
_COLORS = {"black", "white", "blue", "red", "green", "grey", "gray", "pink", "brown", "yellow", "silver", "gold", "navy", "purple"}
_PRODUCT_ID = re.compile(r"\bP-\d{4,}\b", re.I)
_REFS = [(re.compile(r"\b(first two|top two|these two|both|pehle do)\b", re.I), [0, 1]),
         (re.compile(r"\b(first|1st|pehla|pehli)\b(?!\s+two)", re.I), [0]),
         (re.compile(r"\b(second|2nd|dusra|dusri)\b", re.I), [1]),
         (re.compile(r"\b(third|3rd|teesra|teesri)\b", re.I), [2])]
_COMPARE = re.compile(r"\bcompare\b|\bvs\.?\b|\bversus\b|\bdifference between\b|\bmukabla\b", re.I)
_CHEAPER = re.compile(r"\bcheaper\b|\bless expensive\b|\bsaste\b|\bkam (?:daam|price|keemat)\b", re.I)
_FACT = re.compile(r"\b(in stock|available|how much (?:is|does)|price of|kitne ka|kitni keemat)\b", re.I)
_GREETING = re.compile(r"^\s*(hi+|hello|hey|namaste|namaskar)\b[\s!.]*$", re.I)
# Services and topics a shop assistant must decline (answered with a polite refusal)
_OUT_OF_SCOPE = re.compile(
    r"\b(weather|cricket (?:score|match)|who will win|election|politics|joke|movie|recipe|insurance|loan|"
    r"credit card|mutual fund|crypto|share market|stock market|flights?|air ?tickets?|train tickets?|"
    r"bus tickets?|hotel|book (?:me )?(?:a )?(?:flight|ticket|hotel|cab|taxi)|cab|taxi|doctor|medicine)\b", re.I)
# Goods this shop does not sell at all (answered with not_found, no product search)
# (ambiguous words such as "flat" or "house" need context: "flat feet" and "house slippers" are products)
_NOT_SOLD = re.compile(r"\b(electric cars?|cars?|motorcycles?|motorbikes?|scooters?|apartments?|real estate|"
                       r"land plots?|(?:flats?|houses?|plots?) (?:for|to) (?:rent|sale|buy))\b", re.I)


def _to_int(num: str, unit: Optional[str]) -> int:
    value = float(num.replace(",", ""))
    return int(value * 1000) if unit else int(value)


def detect_language(text: str) -> str:
    if _DEVANAGARI.search(text):
        return "hi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    return "hinglish" if len(words & _HINGLISH_WORDS) >= 2 else "en"


def translate_terms(text: str) -> tuple[str, dict[str, str]]:
    """Replace Hindi / Hinglish product words with English catalogue words. Returns (text, mapping)."""
    found: dict[str, str] = {}
    for hi, en in sorted(_HINDI_TERMS.items(), key=lambda kv: -len(kv[0])):   # longest phrase first
        if hi in text:
            text = text.replace(hi, f" {en} ")
            found[hi] = en
    for word, en in _HINGLISH_TERMS.items():
        if re.search(rf"\b{word}\b", text, re.I):
            text = re.sub(rf"\b{word}\b", en, text, flags=re.I)
            found[word] = en
    return re.sub(r"\s+", " ", text).strip(), found


def correct_typos(text: str, vocabulary: Iterable[str], brands: Iterable[str]) -> tuple[str, dict[str, str]]:
    """Fix misspelt product words and brands using the catalogue's own words.

    Only words of 4+ letters that are not already known are changed; the candidate must start with the
    same letter and be within Damerau-Levenshtein distance 1 (2 for words of 7+ letters, e.g. "blutooth").
    """
    vocab = {v.lower() for v in vocabulary}
    brand_map = {b.lower(): b for b in brands if b}
    candidates = vocab | set(brand_map)
    fixes: dict[str, str] = {}

    def fix(m: re.Match) -> str:
        word = m.group(0)
        low = word.lower()
        if len(low) < 4 or low in candidates or low in _KEEP or low in _COLORS:
            return word
        limit = 1 if len(low) <= 6 else 2
        best, best_d = None, limit + 1
        for c in candidates:
            if c[0] != low[0] or abs(len(c) - len(low)) > limit:
                continue
            d = DamerauLevenshtein.distance(low, c)
            if d < best_d or (d == best_d and best is not None and c < best):
                best, best_d = c, d
        if best is None:
            return word
        out = brand_map.get(best, best)
        fixes[word] = out
        return out

    return re.sub(r"[A-Za-z]+", fix, text), fixes


def expand_for_search(text: str) -> str:
    """Add catalogue synonyms (phone -> smartphones, watch -> watches) to the text used for ranking."""
    extra = [_SEARCH_SYNONYMS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in _SEARCH_SYNONYMS]
    return f"{text} {' '.join(extra)}".strip() if extra else text


def parse(message: str, known_brands: Iterable[str] = (), vocabulary: Iterable[str] = ()) -> Slots:
    original = message.strip()
    text = original.translate(_DEV_DIGITS)
    s = Slots(language=detect_language(original))
    brands = {b.lower(): b for b in known_brands if b}

    # budget first, on the untranslated text (Hindi price phrases need the Hindi words)
    if m := _BETWEEN.search(text):
        s.budget_min, s.budget_max = _to_int(m.group(1), m.group(2)), _to_int(m.group(3), m.group(4))
    else:
        for p in _MAX:
            if m := p.search(text):
                s.budget_max = _to_int(m.group(1), m.group(2))
                break
        for p in _MIN:
            if m := p.search(text):
                s.budget_min = _to_int(m.group(1), m.group(2))

    text, translated = translate_terms(text)
    before_typos = text
    text, typos = correct_typos(text, vocabulary, brands.values()) if (vocabulary or brands) else (text, {})
    s.corrections = {**translated, **typos}
    s.query_text = text

    for p in _NEG:
        for word in p.findall(text):
            if word.lower() in brands and brands[word.lower()] not in s.brands_exclude:
                s.brands_exclude.append(brands[word.lower()])
    tokens = re.findall(r"[a-z][\w&-]+", text.lower())
    only = {w.lower() for w in _ONLY.findall(text)}
    for t in tokens:
        if t in brands and brands[t] not in s.brands_exclude and brands[t] not in s.brands_include:
            if only and t not in only:
                continue
            s.brands_include.append(brands[t])

    if m := _SIZE.search(text):
        s.size = (m.group(1) or m.group(2)).upper()
    s.color = next((t for t in tokens if t in _COLORS), None)
    s.product_ids = [pid.upper() for pid in _PRODUCT_ID.findall(text)]
    for pattern, idx in _REFS:
        if pattern.search(text):
            s.references = idx
            break

    # words that describe a product (not brands, negation, filler or numbers)
    product_words = [t for t in tokens if t not in brands and t not in _NEG_WORDS and t not in _KEEP and not t.isdigit()]

    if _GREETING.match(original):
        s.intent = "greeting"
    elif _OUT_OF_SCOPE.search(before_typos):
        s.intent = "out_of_scope"
    elif _NOT_SOLD.search(before_typos):
        s.intent = "not_sold"
    elif _COMPARE.search(text):
        s.intent = "compare"
    elif (s.product_ids or s.references) and _FACT.search(text):
        s.intent = "product_fact"
    elif _CHEAPER.search(text) and not product_words:
        s.wants_cheaper, s.intent = True, "refine"
    elif _CHEAPER.search(text):
        s.wants_cheaper = True          # "cheaper bluetooth speakers": a new search, sorted cheapest first
    elif s.references or ((s.brands_include or s.brands_exclude or s.size or s.color) and not product_words):
        s.intent = "refine"             # "only Nike", "not boAt", "size 9": narrows the previous results

    found = any([s.budget_min, s.budget_max, s.brands_include, s.brands_exclude, s.size, s.color, s.references,
                 s.product_ids, translated])
    s.confidence = 0.8 if found or s.intent != "search" else (0.6 if len(tokens) >= 2 else 0.4)
    return s
