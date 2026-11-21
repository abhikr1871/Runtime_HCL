"""Step 1: safety checks before anything else runs. Owner: M3.

- Requests for another shopper's data are refused without revealing whether that person exists.
- Injection attempts are FLAGGED (recorded in the trace) but the message is still treated as data;
  the LLM has no tools, so obeying an injection cannot change what code returns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.observe.redact import redact

_CUSTOMER_ID = re.compile(r"\bC-\d{3,}\b", re.I)
_OTHER_SHOPPER = [
    re.compile(r"\bwhat (?:did|has|have|does)\s+(?!i\b|me\b|we\b)[a-z]+\s+(?:buy|bought|order|ordered|purchase|purchased)", re.I),
    re.compile(r"\b(?:order|purchase|buying)\s+history\s+of\b", re.I),
    re.compile(r"\b(?:another|other|some)\s+(?:customer|shopper|user)'?s?\b", re.I),
    re.compile(r"\b[a-z]+\s+ne\s+kya\s+(?:kharida|order\s+kiya|liya)", re.I),
    # someone else's personal details: "Rahul's address", "her phone number", "customer 123's preferences"
    re.compile(r"\b(?!my\b|our\b)[a-z]+'s\s+(?:home\s+|saved\s+)?(?:address|phone\s+number|mobile\s+number|number|"
               r"email|contact|orders?|"
               r"order\s+history|purchases?|details|preferences|cart|wishlist|payment|card)", re.I),
    re.compile(r"\b(?:his|her|their)\s+(?:address|phone|mobile\s+number|email|orders?|order\s+history|purchases)", re.I),
    re.compile(r"\b(?:customer|shopper|user|client)\s*(?:id\s*)?#?\s*\d+", re.I),
    re.compile(r"\b(?:last|previous|other|another|every|all)\s+(?:customer|shopper|user)s?'?s?\b", re.I),
    re.compile(r"\b(?:list|show|give|export|dump)\b.{0,30}\b(?:all|every|everyone'?s)\b.{0,20}"
               r"\b(?:orders?|shoppers?|customers?|users?|addresses|records)\b", re.I),
    re.compile(r"\bcustomer\s+records?\b|\bshopper\s+data\b|\buser\s+data(?:base)?\b", re.I),
    # role claims in chat never grant access: "I'm the admin", "admin mode"
    re.compile(r"\b(?:i\s*'?\s*a?m|i\s+am)\s+(?:the\s+|an\s+)?(?:admin|administrator|owner|manager|developer)\b", re.I),
    re.compile(r"\badmin\s+mode\b|\bgod\s+mode\b|\bdeveloper\s+mode\b", re.I),
    # Hindi script: "राहुल ने क्या खरीदा", "उसका पता"
    re.compile(r"ने\s+क्या\s+(?:खरीदा|ख़रीदा|मंगाया|ऑर्डर)"),
    re.compile(r"(?:का|की|के)\s+(?:पता|फ़ोन\s+नंबर|फोन\s+नंबर|मोबाइल\s+नंबर|ऑर्डर)"),
]
# An injection that also reaches for private data is refused outright (it is otherwise just flagged)
_PRIVATE_DATA = re.compile(r"\b(?:orders?|shoppers?|customers?|addresses|phone\s+numbers?|records|passwords?|"
                           r"emails?|database|users?)\b", re.I)
_INJECTION = [
    re.compile(r"ignore\s+(?:all\s+|any\s+|your\s+|the\s+)?(?:previous\s+|prior\s+)?(?:instructions|rules)", re.I),
    re.compile(r"\bsystem\s*:", re.I),
    re.compile(r"\byou are now\b", re.I),
    re.compile(r"\bdisregard\s+(?:your|the|all)\s+(?:rules|instructions)", re.I),
    re.compile(r"\breveal\s+(?:your\s+)?(?:system\s+)?prompt", re.I),
]


@dataclass
class GuardResult:
    refuse: bool = False
    reason: Optional[str] = None
    injection: bool = False
    redacted_message: str = ""


def check(message: str, customer_id: str) -> GuardResult:
    res = GuardResult(redacted_message=redact(message))
    res.injection = any(p.search(message) for p in _INJECTION)

    mentioned = {m.upper() for m in _CUSTOMER_ID.findall(message)}
    if mentioned - {customer_id.upper()}:
        res.refuse, res.reason = True, "other_shopper_data"
    elif any(p.search(message) for p in _OTHER_SHOPPER):
        res.refuse, res.reason = True, "other_shopper_data"
    elif res.injection and _PRIVATE_DATA.search(message):
        res.refuse, res.reason = True, "other_shopper_data"
    return res


_ANY_CUSTOMER_ID = re.compile(r"\bC-\d{3,}\b")


def scrub_output(answer: str, customer_id: str) -> str:
    """Output guard: a reply must never carry ANOTHER shopper's ID (e.g. copied from a review or product text)."""
    return _ANY_CUSTOMER_ID.sub(lambda m: m.group(0) if m.group(0).upper() == customer_id.upper() else "[hidden]", answer)
