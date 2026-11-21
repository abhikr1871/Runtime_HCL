"""Mask PII before anything is logged or stored. Owner: M3."""
from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")


def redact(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _CARD.sub("[number]", text)
    return _PHONE.sub("[phone]", text)
