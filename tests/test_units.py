"""Unit tests for deterministic pieces: understanding rules, normaliser, validator, fusion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from app.pipeline import nlu_rules, validator  # noqa: E402
from app.pipeline.reasons import format_inr  # noqa: E402
from app.pipeline.retrieval import rrf_fuse  # noqa: E402
from app.schemas import Product  # noqa: E402
from app.store.normalise import parse_price_inr, weight_to_grams  # noqa: E402


def test_budget_phrases():
    assert nlu_rules.parse("shoes under 3k").budget_max == 3000
    assert nlu_rules.parse("speaker ₹2,000 se kam").budget_max == 2000
    s = nlu_rules.parse("between 1000 and 2500")
    assert (s.budget_min, s.budget_max) == (1000, 2500)


def test_language_detection():
    assert nlu_rules.detect_language("mujhe sasta speaker chahiye") == "hinglish"
    assert nlu_rules.detect_language("मुझे जूते चाहिए") == "hi"
    assert nlu_rules.detect_language("running shoes for flat feet") == "en"


def test_brand_negation_and_only():
    s = nlu_rules.parse("running shoes not Nike", known_brands=["Nike", "Campus"])
    assert s.brands_exclude == ["Nike"] and s.brands_include == []
    s = nlu_rules.parse("only Campus", known_brands=["Nike", "Campus"])
    assert s.brands_include == ["Campus"]


def test_weight_normalisation():
    assert weight_to_grams("1.2 kg") == 1200
    assert weight_to_grams("500g") == 500
    assert weight_to_grams("540") is None  # no unit: never guess
    assert parse_price_inr("Rs. 2,499") == 2499


def test_indian_number_format():
    assert format_inr(120000) == "₹1,20,000"
    assert format_inr(2499) == "₹2,499"


def test_validator_catches_wrong_price():
    p = Product(product_id="P-0001", name="X", category="c", price_inr=2499, stock=3)
    assert validator.check("X (P-0001) costs ₹2,499", [p]).ok
    assert not validator.check("X (P-0001) costs ₹1,999", [p]).ok
    assert not validator.check("Try P-0002 instead", [p]).ok
    assert validator.check("under your ₹3,000 budget", [p], allowed_amounts=[3000]).ok


def test_rrf_fusion():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
    assert [d for d, _ in fused][:2] in (["a", "b"], ["b", "a"])
    assert dict(fused)["d"] < dict(fused)["a"]
