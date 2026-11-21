"""Step 2a: rules-first understanding (fast, no LLM). Owner: M1.

Extracts language, budget, brands (incl. negation), size, colour, references and intent.
The LLM (nlu_llm.py) is only asked when these rules leave the request unclear.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from app.schemas import Slots

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_HINGLISH_WORDS = {
    "mujhe", "chahiye", "sasta", "sasti", "saste", "kitna", "kitne", "hai", "kya", "ka", "ki", "ke", "se",
    "kam", "wala", "wali", "dikhao", "acha", "accha", "nahi", "nahin", "tak", "mein", "aur", "bhi", "liye", "koi",
}
_NUM = r"(?:rs\.?|inr|₹)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b|thousand|hazaar|hazar)?"
_MAX = [re.compile(r"(?:under|below|less than|within|up\s?to|max(?:imum)?|budget(?:\s+of|\s+is)?)\s*" + _NUM, re.I),
        re.compile(_NUM + r"\s*(?:se\s+kam|ke\s+andar|ke\s+neeche|tak|or less)", re.I)]
_MIN = [re.compile(r"(?:above|over|more than|at least|min(?:imum)?)\s*" + _NUM, re.I)]
_BETWEEN = re.compile(r"between\s*" + _NUM + r"\s*(?:and|to|-)\s*" + _NUM, re.I)
_NEG = [re.compile(r"\b(?:not|no|except|without|but not)\s+([a-z][\w&-]+)", re.I),
        re.compile(r"\b([a-z][\w&-]+)\s+(?:nahi|nahin|mat)\b", re.I)]
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
_OUT_OF_SCOPE = re.compile(r"\b(weather|cricket score|election|politics|joke|movie|recipe)\b", re.I)


def _to_int(num: str, unit: Optional[str]) -> int:
    value = float(num.replace(",", ""))
    return int(value * 1000) if unit else int(value)


def detect_language(text: str) -> str:
    if _DEVANAGARI.search(text):
        return "hi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    return "hinglish" if len(words & _HINGLISH_WORDS) >= 2 else "en"


def parse(message: str, known_brands: Iterable[str] = ()) -> Slots:
    text = message.strip()
    s = Slots(query_text=text, language=detect_language(text))
    brands = {b.lower(): b for b in known_brands if b}

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

    for p in _NEG:
        for word in p.findall(text):
            if word.lower() in brands:
                s.brands_exclude.append(brands[word.lower()])
    tokens = re.findall(r"[a-z][\w&-]+", text.lower())
    only = {w.lower() for w in _ONLY.findall(text)}
    for t in tokens:
        if t in brands and brands[t] not in s.brands_exclude and brands[t] not in s.brands_include:
            if only and t not in only:
                continue
            s.brands_include.append(brands[t])
    # TODO(M1): typo tolerance with rapidfuzz ("nkie" -> Nike) against known_brands

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
