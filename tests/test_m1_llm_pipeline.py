"""Unit tests for M1 Tasks 2 & 3: LLM JSON retry, stock validation, and regeneration."""
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from app.config import get_settings  # noqa: E402
from app.pipeline import composer, nlu_llm, orchestrator, validator  # noqa: E402
from app.schemas import ChatRequest, Product, Slots  # noqa: E402
from app.store import db, indexer  # noqa: E402


class FakeLLMClient:
    """Mock-free fake LLM client for deterministic pipeline tests without network calls."""

    def __init__(
        self,
        json_responses: Optional[list[Any]] = None,
        text_responses: Optional[list[Optional[str]]] = None,
    ) -> None:
        self.json_responses = list(json_responses or [])
        self.text_responses = list(text_responses or [])
        self.json_calls = 0
        self.text_calls = 0
        self.calls = 0
        self.mock = False
        self.name = "fake-llm"
        self.tokens_in = 0
        self.tokens_out = 0

    def complete_json(self, system: str, user: str, **kwargs) -> Any:
        self.json_calls += 1
        self.calls += 1
        if self.json_responses:
            return self.json_responses.pop(0)
        return None

    def complete_text(self, system: str, user: str, **kwargs) -> Optional[str]:
        self.text_calls += 1
        self.calls += 1
        if self.text_responses:
            return self.text_responses.pop(0)
        return None


# ---------------------------------------------------------------------------
# (a) Invalid JSON then valid JSON -> retry used
# ---------------------------------------------------------------------------
def test_nlu_llm_retry_on_invalid_json():
    rules = Slots(query_text="shoes", confidence=0.4)
    # First response is invalid (None), second response is valid JSON
    fake_llm = FakeLLMClient(json_responses=[None, {"intent": "search", "budget_max": 2500}])

    refined = nlu_llm.refine("shoes", rules, fake_llm)
    assert fake_llm.json_calls == 2, "Expected exactly 2 LLM calls (initial + 1 retry)"
    assert refined.budget_max == 2500
    assert refined.source == "rules+llm"


def test_nlu_llm_rules_win_on_budget_and_brand():
    rules = Slots(
        query_text="running shoes",
        budget_max=3000,
        brands_include=["Nike"],
        confidence=0.4,
    )
    # LLM contradicts rules with different budget and brand
    fake_llm = FakeLLMClient(json_responses=[{"budget_max": 5000, "brands_include": ["Adidas"], "color": "blue"}])

    refined = nlu_llm.refine("running shoes", rules, fake_llm)
    assert fake_llm.json_calls == 1
    assert refined.budget_max == 3000, "Rules budget must not be overwritten by LLM"
    assert refined.brands_include == ["Nike"], "Rules brand must not be overwritten by LLM"
    assert refined.color == "blue", "New attributes from LLM should be applied"


# ---------------------------------------------------------------------------
# (b) Invalid twice -> rules fallback
# ---------------------------------------------------------------------------
def test_nlu_llm_invalid_twice_falls_back_to_rules():
    rules = Slots(query_text="shoes", confidence=0.4)
    # Both initial and retry fail (e.g. invalid JSON or schema failure)
    fake_llm = FakeLLMClient(json_responses=[None, None])

    refined = nlu_llm.refine("shoes", rules, fake_llm)
    assert fake_llm.json_calls == 2, "Expected retry to be attempted once"
    assert refined == rules, "Must fall back to rules when retry fails"


def test_nlu_llm_schema_validation_error_retried():
    rules = Slots(query_text="shoes", confidence=0.4)
    # First response fails Pydantic validation (budget_max as string that can't be parsed)
    fake_llm = FakeLLMClient(json_responses=[{"budget_max": "unlimited"}, {"budget_max": 2000}])

    refined = nlu_llm.refine("shoes", rules, fake_llm)
    assert fake_llm.json_calls == 2
    assert refined.budget_max == 2000


# ---------------------------------------------------------------------------
# (c) Validator stock claim checks
# ---------------------------------------------------------------------------
def test_validator_stock_claims():
    p_out = Product(product_id="P-0001", name="Shoes", category="c", price_inr=2499, stock=0)
    p_in = Product(product_id="P-0002", name="Tent", category="c", price_inr=4999, stock=5)

    # Claiming "in stock" for stock == 0 -> flagged
    res1 = validator.check("Shoes (P-0001) costs ₹2,499 and is in stock", [p_out])
    assert not res1.ok
    assert any("in stock" in p for p in res1.problems)

    # Claiming "available" for stock == 0 -> flagged
    res1b = validator.check("Shoes (P-0001) costs ₹2,499 and is available", [p_out])
    assert not res1b.ok

    # Claiming "only 2 left" when stock == 5 -> flagged
    res2 = validator.check("Tent (P-0002) costs ₹4,999 and only 2 left", [p_in])
    assert not res2.ok
    assert any("claimed 2 left" in p for p in res2.problems)

    # Claiming "out of stock" when stock == 5 -> flagged
    res3 = validator.check("Tent (P-0002) costs ₹4,999 and is out of stock", [p_in])
    assert not res3.ok
    assert any("out of stock" in p for p in res3.problems)

    # Correct claims -> pass
    res_ok = validator.check(
        "Shoes (P-0001) costs ₹2,499 and is out of stock. Tent (P-0002) costs ₹4,999 and is in stock.",
        [p_out, p_in],
    )
    assert res_ok.ok
    assert not res_ok.problems

    # Exact stock match -> pass
    res_exact = validator.check("Tent (P-0002) costs ₹4,999 with only 5 left", [p_in])
    assert res_exact.ok


# ---------------------------------------------------------------------------
# (d) Validator failure -> one regeneration -> then template / accepted
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    db_file = tmp_path / "test_pipeline.db"
    os.environ["DB_PATH"] = str(db_file)
    os.environ["MOCK_LLM"] = "false"
    os.environ["ENABLE_EMBEDDINGS"] = "false"
    get_settings.cache_clear()
    db.reset_conn()
    db.init_db()
    db.seed_sample_if_empty()
    indexer.rebuild()
    yield
    db.reset_conn()


def test_orchestrator_regeneration_fixes_validation(monkeypatch):
    # First response has invented price ₹99,999 (fails validator).
    # Second response (regeneration) fixes price to ₹2,999 (valid catalogue price for P-9002).
    fake_llm = FakeLLMClient(
        text_responses=[
            "Here are options: Summit Trekking Shoes (P-9002) costs ₹99,999 and is in stock.",
            "Here are options: Summit Trekking Shoes (P-9002) costs ₹2,999 and is in stock.",
        ]
    )
    monkeypatch.setattr(orchestrator, "LLMClient", lambda: fake_llm)

    req = ChatRequest(session_id="regen_fix", customer_id="C-001", message="trekking shoes under 3000")
    resp = orchestrator.handle_chat(req)

    assert resp.answer_type == "recommendation"
    assert "₹2,999" in resp.answer
    assert fake_llm.text_calls == 2, "Expected composer to be called twice (initial + regeneration)"


def test_orchestrator_regeneration_failure_falls_back_to_template(monkeypatch):
    # Both initial and regenerated responses contain invalid price ₹99,999.
    fake_llm = FakeLLMClient(
        text_responses=[
            "Summit Trekking Shoes (P-9002) costs ₹99,999.",
            "Still Summit Trekking Shoes (P-9002) costs ₹99,999.",
        ]
    )
    monkeypatch.setattr(orchestrator, "LLMClient", lambda: fake_llm)

    req = ChatRequest(session_id="regen_fail", customer_id="C-001", message="trekking shoes under 3000")
    resp = orchestrator.handle_chat(req)

    assert resp.answer_type == "recommendation"
    assert fake_llm.text_calls == 2, "Expected composer to be called twice"
    # Should fall back to template containing valid items
    assert "Here are" in resp.answer
    assert "₹2,999" in resp.answer
