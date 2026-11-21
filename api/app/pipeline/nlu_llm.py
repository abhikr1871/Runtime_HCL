"""Step 2b: LLM fallback for understanding. Owner: M1.

Called only when rules are unsure (low confidence). Output must validate against `Slots`;
on any failure we keep the rules result, so the shopper still gets an answer.
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
        data = llm.complete_json(_PROMPT, message)
        from_llm, err = _try_parse_slots(data)

        if from_llm is None:
            # Retry once, showing the error message in the prompt
            retry_prompt = f"{message}\n\nYour previous output was invalid: {err}. Return only valid JSON matching the schema."
            retry_data = llm.complete_json(_PROMPT, retry_prompt)
            from_llm, retry_err = _try_parse_slots(retry_data)
            if from_llm is None:
                log.info("LLM refine retry failed: %s; falling back to rules", retry_err)
                return rules

        clean = {k: getattr(from_llm, k) for k in Slots.model_fields if getattr(from_llm, k) not in (None, "", [])}
        update = {k: v for k, v in clean.items() if k not in {"query_text", "references", "product_ids"}}

        # Rules win on budget numbers if rules found one
        if rules.budget_max is not None:
            update["budget_max"] = rules.budget_max
        if rules.budget_min is not None:
            update["budget_min"] = rules.budget_min

        # Rules win on brands if rules found explicit matches
        if rules.brands_include:
            update["brands_include"] = rules.brands_include
        if rules.brands_exclude:
            update["brands_exclude"] = rules.brands_exclude

        return rules.model_copy(update={**update, "source": "rules+llm"})
    except Exception as exc:
        log.warning("nlu_llm refine error: %s; falling back to rules", exc)
        return rules
