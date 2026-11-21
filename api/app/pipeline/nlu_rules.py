"""Step 2a: rules-first understanding (fast, no LLM). Owner: M1.

Turns the shopper's message into Slots:
- language (English / Hindi in Devanagari / Hinglish)
- Hindi and Hinglish product words -> English search words ("कुर्ता" -> kurta, "joote" -> shoes)
- typo correction against the catalogue's own vocabulary ("blutooth" -> bluetooth, "nkie" -> Nike)
- budget in English, Hinglish and Hindi ("under 3k", "2000 se kam", "2000 रुपये से कम", "3 hazaar tak",
  number words "paanch hazaar" / "पाँच सौ", lakh, minimums "2000 se zyada", ranges "2000 se 4000 ke beech")
- brands to include / exclude, typo-tolerant and multi-word ("only Nike", "not boAt", "nkie nahi", "Peter England")

Budget-phrase parsing and the brand matcher were contributed by Daksh (M1) and combined with the rest here.
- size, colour, explicit product IDs, references ("first two", "the second one")
- intent: search, refine, compare, product_fact, greeting, out_of_scope, not_sold

The LLM (nlu_llm.py) is only asked when these rules leave the request unclear.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from rapidfuzz import distance, fuzz
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

# ---------- budget phrases (Daksh) ----------
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
    "k": 1000, "thousand": 1000, "hazaar": 1000, "hazar": 1000, "hajar": 1000, "हज़ार": 1000, "हजार": 1000,
    "sau": 100, "सौ": 100, "lakh": 100000, "lac": 100000, "लाख": 100000,
}
_WORDS_PAT = "|".join(sorted(_WORDS_TO_NUM, key=lambda x: -len(x)))
_MULT_PAT = r"(?:k\b|thousand|hazaar|hazar|hajar|हज़ार|हजार|sau|सौ|lakh|lac|लाख)"
_CURRENCY_WORD = r"(?:\s*(?:rs\.?|rupees|rupaye|rupay|रुपये|रुपए|रुपया|रु\.?))?"
_NUM = (
    r"(?:(?:rs\.?|inr|₹)\s*)?"
    r"(?:(\d[\d,]*(?:\.\d+)?)|(?<![a-z])(" + _WORDS_PAT + r")(?![a-z]))"   # number words must stand alone ("do", not "dove")
    r"(?:\s*(" + _MULT_PAT + r"))?"
    r"(?!\s*(?:gb|tb|mb|mp|g\b|inch|hz|mah))"                             # "32 GB", "5G" are specs, not prices
)
_BETWEEN = [
    re.compile(r"between\s*" + _NUM + r"\s*(?:and|to|-)\s*" + _NUM, re.I),
    re.compile(_NUM + _CURRENCY_WORD + r"\s*(?:se|to|-|से)\s*" + _NUM + _CURRENCY_WORD +
               r"\s*(?:ke\s+(?:beech|bich)|के\s*बीच|tak|तक)", re.I),
]
_MAX = [
    re.compile(r"(?:under|below|less than|within|up\s?to|upto|max(?:imum)?|budget(?:\s+of|\s+is)?)\s*" + _NUM, re.I),
    re.compile(_NUM + _CURRENCY_WORD + r"\s*(?:se\s+kam|से\s*कम|ke\s+andar|के\s*अंदर|ke\s+neeche|के\s*नीचे|"
               r"tak(?:\s+ka)?|तक(?:\s*का)?|or less|budget)", re.I),
    re.compile(r"(?:rs\.?|inr|₹)\s*" + _NUM, re.I),
]
_MIN = [
    re.compile(r"(?:above|over|more than|at least|min(?:imum)?|starting(?:\s+from)?|start(?:\s+from)?)\s*" + _NUM, re.I),
    re.compile(_NUM + _CURRENCY_WORD + r"\s*(?:se\s+(?:zyada|jyada|upar|adhik)|से\s*(?:ज़्यादा|ज्यादा|ऊपर|अधिक)|or more)", re.I),
]
# Words that must never be read as a brand typo (Daksh)
_BRAND_STOPWORDS = {
    "shoes", "shoe", "phone", "phones", "under", "below", "black", "white", "blue", "green", "red",
    "price", "show", "best", "cheap", "cheaper", "gift", "gifts", "good", "laptop", "laptops",
    "watch", "watches", "shirt", "shirts", "sound", "size", "with", "from", "that", "this",
    "free", "deal", "more", "less", "sale", "home", "wear", "mens", "womens", "boys", "girls",
    "chahiye", "dikhao", "sasta", "saste", "accha", "acha", "wala", "wali", "wale", "karo", "karna",
    "only", "nahi", "nahin", "mat", "not", "between", "over", "above", "item", "items", "both",
}
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


def _parse_val(num_str: Optional[str], word_str: Optional[str], mult_str: Optional[str]) -> Optional[int]:
    """'3' + 'hazaar' -> 3000, 'paanch' + 'sau' -> 500, '1.5' + 'lakh' -> 150000 (Daksh)."""
    if num_str:
        val = float(num_str.replace(",", ""))
    elif word_str:
        val = float(_WORDS_TO_NUM[word_str.lower()])
    else:
        return None
    if mult_str:
        val *= _MULT.get(mult_str.lower(), 1)
    return int(val)


def _match_brand(token: str, brands: dict[str, str], threshold: float = 85.0) -> Optional[str]:
    """Known brand for a token: exact match first, then typo-tolerant rapidfuzz (Daksh)."""
    t = token.lower()
    if t in brands:
        return brands[t]
    if len(t) < 4 or t in _BRAND_STOPWORDS:
        return None
    best_brand, best_score = None, 0.0
    for b_lower, b_orig in brands.items():
        if " " in b_lower:
            continue
        ratio = fuzz.ratio(t, b_lower)
        jw = distance.JaroWinkler.similarity(t, b_lower) * 100.0
        dl = distance.DamerauLevenshtein.distance(t, b_lower)
        score = max(ratio, jw) if (dl <= 1 and abs(len(t) - len(b_lower)) <= 1) else ratio
        if score >= threshold and score > best_score:
            best_score, best_brand = score, b_orig
    return best_brand


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
        if len(low) < 4 or low in candidates or low in _KEEP or low in _COLORS or low in _BRAND_STOPWORDS:
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
    range_found = False
    for p in _BETWEEN:
        if m := p.search(text):
            s.budget_min = _parse_val(m.group(1), m.group(2), m.group(3))
            s.budget_max = _parse_val(m.group(4), m.group(5), m.group(6))
            range_found = True
            break
    if not range_found:
        for p in _MAX:
            if m := p.search(text):
                s.budget_max = _parse_val(m.group(1), m.group(2), m.group(3))
                break
        for p in _MIN:
            if m := p.search(text):
                s.budget_min = _parse_val(m.group(1), m.group(2), m.group(3))
                break

    text, translated = translate_terms(text)
    before_typos = text
    text, typos = correct_typos(text, vocabulary, brands.values()) if (vocabulary or brands) else (text, {})
    s.corrections = {**translated, **typos}
    s.query_text = text

    # brands (Daksh): multi-word brands, then negation / "only" / mentions, each typo-tolerant
    text_lower = text.lower()
    for b_lower, b_orig in brands.items():
        if " " in b_lower and b_lower in text_lower:
            neg = (r"\b(?:not|no|except|without|but not)\s+" + re.escape(b_lower) + r"\b|"
                   r"\b" + re.escape(b_lower) + r"\s+(?:nahi|nahin|mat)\b")
            if re.search(neg, text_lower):
                if b_orig not in s.brands_exclude:
                    s.brands_exclude.append(b_orig)
            elif b_orig not in s.brands_include:
                s.brands_include.append(b_orig)
    for p in _NEG:
        for word in p.findall(text):
            if (matched := _match_brand(word, brands)) and matched not in s.brands_exclude:
                s.brands_exclude.append(matched)
    only_matched = {m for w in _ONLY.findall(text) if (m := _match_brand(w, brands))}
    tokens = re.findall(r"[a-z][\w&-]+", text_lower)
    for t in tokens:
        matched = _match_brand(t, brands)
        if matched and matched not in s.brands_exclude and matched not in s.brands_include:
            if only_matched and matched not in only_matched:
                continue
            s.brands_include.append(matched)
    brand_words = {w for b in brands for w in b.split()}

    if m := _SIZE.search(text):
        s.size = (m.group(1) or m.group(2)).upper()
    s.color = next((t for t in tokens if t in _COLORS), None)
    s.product_ids = [pid.upper() for pid in _PRODUCT_ID.findall(text)]
    for pattern, idx in _REFS:
        if pattern.search(text):
            s.references = idx
            break

    # words that describe a product (not brands, negation, filler or numbers)
    product_words = [t for t in tokens if t not in brands and t not in brand_words and t not in _NEG_WORDS
                     and t not in _KEEP and not t.isdigit() and not _match_brand(t, brands)]

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
