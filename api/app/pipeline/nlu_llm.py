"""Step 2b: LLM fallback for understanding. Owner: M1.

Called only when rules are unsure (low confidence). Output must validate against `Slots`;
on any failure we keep the rules result, so the shopper still gets an answer.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from app.ai.llm_client import LLMClient
from app.schemas import Slots

_PROMPT = (Path(__file__).resolve().parents[1] / "ai" / "prompts" / "nlu.md").read_text(encoding="utf-8")
CONFIDENCE_THRESHOLD = 0.6


def refine(message: str, rules: Slots, llm: LLMClient) -> Slots:
    if rules.confidence >= CONFIDENCE_THRESHOLD:
        return rules
    data = llm.complete_json(_PROMPT, message)
    if not data:
        return rules
    try:
        clean = {k: v for k, v in data.items() if k in Slots.model_fields and v not in (None, "", [])}
        from_llm = Slots(**clean)
    except ValidationError:
        # TODO(M1): retry once, showing the model the validation error
        return rules
    update = {k: getattr(from_llm, k) for k in clean if k not in {"query_text", "references", "product_ids"}}
    return rules.model_copy(update={**update, "source": "rules+llm"})
