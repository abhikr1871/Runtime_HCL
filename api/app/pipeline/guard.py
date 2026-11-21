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
]
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
    # TODO(M3): more phrasings (Hindi script), and an output guard that scans replies for other customers' IDs/names
    return res
