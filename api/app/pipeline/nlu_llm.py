"""Step 2b: LLM fallback for understanding. Owner: M1.

Called only when rules are unsure (low confidence). Output must validate against `Slots`;
an invalid answer is retried once with the validation error shown to the model (Daksh), and on any
failure we keep the rules result, so the shopper still gets an answer. Numbers and brands the rules
already found always win over the LLM.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.ai.llm_client import LLMClient
from app.schemas import Slots

log = logging.getLogger("nlu_llm")
_PROMPT = (Path(__file__).resolve().parents[1] / "ai" / "prompts" / "nlu.md").read_text(encoding="utf-8")
CONFIDENCE_THRESHOLD = 0.6
MAX_TOKENS = 600   # enough for the JSON (reasoning models think first); keeps calls inside the rate limit


def _try_parse_slots(data: Any) -> tuple[Slots | None, str | None]:
    if not isinstance(data, dict):
        return None, "Output was not a JSON object or was empty"
    try:
        clean = {k: v for k, v in data.items() if k in Slots.model_fields and v not in (None, "", [])}
        return Slots(**clean), None
    except ValidationError as err:
        return None, str(err)
    except Exception as err:
        return None, str(err)


def refine(message: str, rules: Slots, llm: LLMClient) -> Slots:
    if rules.confidence >= CONFIDENCE_THRESHOLD:
        return rules
    try:
        data = llm.complete_json(_PROMPT, message, max_tokens=MAX_TOKENS)
        from_llm, err = _try_parse_slots(data)
        if from_llm is None:
            # Retry once, showing the validation error to the model
            retry_prompt = f"{message}\n\nYour previous output was invalid: {err}. Return only valid JSON matching the schema."
            from_llm, retry_err = _try_parse_slots(llm.complete_json(_PROMPT, retry_prompt, max_tokens=MAX_TOKENS))
            if from_llm is None:
                log.info("LLM refine retry failed: %s; falling back to rules", retry_err)
        if from_llm is None:
            return rules

        clean = {k: getattr(from_llm, k) for k in Slots.model_fields if getattr(from_llm, k) not in (None, "", [])}
        update = {k: v for k, v in clean.items() if k not in {"query_text", "references", "product_ids", "corrections"}}
        # Rules win on numbers and explicit brands they already found
        for key in ("budget_max", "budget_min", "brands_include", "brands_exclude", "language"):
            value = getattr(rules, key)
            if value not in (None, [], ""):
                update[key] = value
        return rules.model_copy(update={**update, "source": "rules+llm"})
    except Exception as exc:
        log.warning("nlu_llm refine error: %s; falling back to rules", exc)
        return rules
