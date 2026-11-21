"""Step 7b: word the reply. Owner: M1.

The template is always available; the LLM only rephrases code-built facts, and the validator
checks its output. Catalogue text is passed inside <catalogue_data> as data, never as instructions.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.ai.llm_client import LLMClient
from app.pipeline.reasons import format_inr
from app.schemas import Product

_PROMPT = (Path(__file__).resolve().parents[1] / "ai" / "prompts" / "composer.md").read_text(encoding="utf-8")
_LANG_NAME = {"en": "English", "hi": "Hindi (Devanagari script)", "hinglish": "Hinglish (Hindi in Roman script)"}
_INTRO = {
    "en": "Here are {n} options that fit:",
    "hinglish": "Ye {n} options aapke liye sahi rahenge:",
    "hi": "ये {n} विकल्प आपके लिए सही रहेंगे:",
}


_INTRO_ONE = {
    "en": "Here is an option that fits:",
    "hinglish": "Ye option aapke liye sahi rahega:",
    "hi": "यह विकल्प आपके लिए सही रहेगा:",
}


def template(items: list[tuple[Product, str]], language: str) -> str:
    intro = _INTRO_ONE if len(items) == 1 else _INTRO
    lines = [intro.get(language, intro["en"]).format(n=len(items))]
    for p, reason in items:
        lines.append(f"- {p.name} ({p.product_id}): {reason}")
    return "\n".join(lines)


def compose(items: list[tuple[Product, str]], language: str, llm: LLMClient, extra_instruction: str = "") -> str:
    fallback = template(items, language)
    facts = [{"product_id": p.product_id, "name": p.name, "price": format_inr(p.price_inr), "reason": r} for p, r in items]
    user = f"<catalogue_data>\n{json.dumps(facts, ensure_ascii=False)}\n</catalogue_data>"
    if extra_instruction:
        user += f"\n\nCorrection required: {extra_instruction}\nUse only the exact facts inside <catalogue_data>."
    text = llm.complete_text(_PROMPT.replace("{language}", _LANG_NAME.get(language, "English")), user)
    return text.strip() if text else fallback
