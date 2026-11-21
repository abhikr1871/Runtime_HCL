"""Unit tests for M1 Tasks 4, 5 & 6: smarter clarify, counterfactual relax hints, and matched tags in reasons."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from app.pipeline import clarify, reasons, relax  # noqa: E402
from app.schemas import Product, SessionState  # noqa: E402


def _make_prod(
    pid: str,
    name: str = "P",
    category: str = "mens-shoes",
    price: int = 2000,
    sizes: list[str] = None,
    tags: list[str] = None,
    brand: str = "Nike",
    stock: int = 5,
    rating: float = 4.5,
) -> Product:
    return Product(
        product_id=pid,
        name=name,
        category=category,
        price_inr=price,
        sizes=sizes or [],
        tags=tags or [],
        brand=brand,
        stock=stock,
        rating=rating,
    )


# ---------------------------------------------------------------------------
# Task 4: clarify.py tests
# ---------------------------------------------------------------------------
def test_clarify_picks_highest_variance_attribute():
    # 4 shoe matches where sizes vary widely (4 distinct sizes) while brand is identical ("Nike")
    matches = [
        _make_prod("P-1", sizes=["7"], brand="Nike", price=2000),
        _make_prod("P-2", sizes=["8"], brand="Nike", price=2000),
        _make_prod("P-3", sizes=["9"], brand="Nike", price=2000),
        _make_prod("P-4", sizes=["10"], brand="Nike", price=2000),
    ]
    state = SessionState(session_id="s1", customer_id="C-1", filters={"query_text": "shoes"})
    res = clarify.next_question(state, matches, "en")

    assert res is not None
    slot, q = res
    assert slot == "size"
    assert "7" in q and "8" in q and "size" in q.lower()


def test_clarify_skips_known_filters_and_picks_next_highest():
    # Size is already known in filters; use tags vary (running, trekking, gym)
    matches = [
        _make_prod("P-1", sizes=["9"], tags=["running"], brand="Nike", price=2000),
        _make_prod("P-2", sizes=["9"], tags=["trekking"], brand="Nike", price=2000),
        _make_prod("P-3", sizes=["9"], tags=["gym"], brand="Nike", price=2000),
        _make_prod("P-4", sizes=["9"], tags=["party"], brand="Nike", price=2000),
    ]
    state = SessionState(
        session_id="s1",
        customer_id="C-1",
        filters={"query_text": "shoes", "size": "9"},
    )
    res = clarify.next_question(state, matches, "en")

    assert res is not None
    slot, q = res
    assert slot == "use"
    assert "using it for" in q


def test_clarify_respects_limits():
    matches = [
        _make_prod("P-1", sizes=["7", "8"]),
        _make_prod("P-2", sizes=["9", "10"]),
        _make_prod("P-3", sizes=["11"]),
        _make_prod("P-4", sizes=["12"]),
    ]
    # Limit 1: clarify_streak >= 2 -> stop
    state_max_streak = SessionState(
        session_id="s1",
        customer_id="C-1",
        clarify_streak=2,
        filters={"query_text": "shoes"},
    )
    assert clarify.next_question(state_max_streak, matches, "en") is None

    # Limit 2: matches <= 3 -> never ask
    state_ok = SessionState(session_id="s1", customer_id="C-1", clarify_streak=0, filters={"query_text": "shoes"})
    assert clarify.next_question(state_ok, matches[:3], "en") is None


def test_clarify_language_text():
    matches = [
        _make_prod("P-1", sizes=["7"], price=2000),
        _make_prod("P-2", sizes=["8"], price=2000),
        _make_prod("P-3", sizes=["9"], price=2000),
        _make_prod("P-4", sizes=["10"], price=2000),
    ]
    state = SessionState(session_id="s1", customer_id="C-1", filters={"query_text": "shoes"})

    q_en = clarify.next_question(state, matches, "en")[1]
    assert "What size" in q_en

    q_hinglish = clarify.next_question(state, matches, "hinglish")[1]
    assert "size chahiye" in q_hinglish

    q_hi = clarify.next_question(state, matches, "hi")[1]
    assert "साइज़ चाहिए" in q_hi


# ---------------------------------------------------------------------------
# Task 5: relax.py tests
# ---------------------------------------------------------------------------
def test_relax_counterfactual_budget_hint(monkeypatch):
    # Mock filters.apply to simulate candidate pool above budget
    candidates = [
        _make_prod("P-1", price=2500),
        _make_prod("P-2", price=2500),
        _make_prod("P-3", price=3000),
    ]
    monkeypatch.setattr("app.pipeline.filters.apply", lambda loosened: candidates)
    monkeypatch.setattr("app.pipeline.retrieval.rank", lambda q, c, s: [(p, 1.0) for p in c])

    # Shopper set budget_max = 2000 (blocks all candidates)
    f = {"budget_max": 2000, "category": "mens-shoes"}

    # English check: min price is 2500, diff is 500, count at min price is 2
    res_en = relax.relax(f, "shoes", "en")
    assert res_en is not None
    label, ranked, msg_en, extra = res_en
    assert label == "budget"
    assert "₹500 more (₹2,500) gives 2 options." in msg_en
    assert extra == [500, 2500, 2000]
    assert len(ranked) == 2
    assert all(p.price_inr <= 2500 for p, _ in ranked)

    # Hinglish check
    res_hinglish = relax.relax(f, "shoes", "hinglish")
    assert "₹500 zyada (₹2,500) mein 2 options milenge." in res_hinglish[2]

    # Hindi check
    res_hi = relax.relax(f, "shoes", "hi")
    assert "₹500 अधिक (₹2,500) में 2 विकल्प उपलब्ध हैं।" in res_hi[2]


def test_relax_budget_fixtures_card_consistency(monkeypatch):
    # Test with fixtures: budget 3000, products at 3500, 3500, 8000 -> hint says 2 options and ranked list has only two 3500 items
    candidates = [
        _make_prod("P-1", price=3500),
        _make_prod("P-2", price=3500),
        _make_prod("P-3", price=8000),
    ]
    monkeypatch.setattr("app.pipeline.filters.apply", lambda loosened: candidates)
    monkeypatch.setattr("app.pipeline.retrieval.rank", lambda q, c, s: [(p, 1.0) for p in c])

    f = {"budget_max": 3000, "category": "mens-shoes"}
    res = relax.relax(f, "shoes", "en")
    assert res is not None
    label, ranked, msg, extra = res
    assert "₹500 more (₹3,500) gives 2 options." in msg
    assert len(ranked) == 2
    assert [p.product_id for p, _ in ranked] == ["P-1", "P-2"]
    assert all(p.price_inr <= 3500 for p, _ in ranked)


def test_relax_non_budget_filter_removal(monkeypatch):
    candidates = [_make_prod("P-1"), _make_prod("P-2"), _make_prod("P-3")]
    monkeypatch.setattr("app.pipeline.filters.apply", lambda loosened: candidates)
    monkeypatch.setattr("app.pipeline.retrieval.rank", lambda q, c, s: [(p, 1.0) for p in c])

    f = {"color": "red", "category": "mens-shoes"}
    res = relax.relax(f, "shoes", "en")
    assert res is not None
    label, ranked, msg, extra = res
    assert label == "colour"
    assert "Removing your colour filter gives 3 options." in msg


# ---------------------------------------------------------------------------
# Task 6: reasons.py tests
# ---------------------------------------------------------------------------
def test_reasons_includes_matched_tags():
    p = _make_prod(
        "P-9002",
        name="Summit Trekking Shoes",
        price=2999,
        rating=4.4,
        stock=6,
        tags=["trekking", "hiking", "shoes", "waterproof"],
    )

    f = {"query_text": "waterproof shoes for trekking under 3000", "budget_max": 3000}
    reason, fields = reasons.build(p, f)

    assert "waterproof" in reason
    assert "for trekking" in reason
    assert "tags" in fields
    assert "₹2,999, under your ₹3,000 budget" in reason


def test_reasons_unmatched_tags_remain_unchanged():
    p = _make_prod(
        "P-9002",
        name="Summit Trekking Shoes",
        price=2999,
        rating=4.4,
        stock=6,
        tags=["trekking", "waterproof"],
    )

    f = {"query_text": "shoes"}
    reason, fields = reasons.build(p, f)

    assert "waterproof" not in reason
    assert "trekking" not in reason
    assert "tags" not in fields


def test_reasons_tag_matching_precision():
    # query "shoes for running" must NOT match tag "for flat feet" but MUST match tag "running"
    p1 = _make_prod("P-1", tags=["for flat feet", "running"])
    reason1, _ = reasons.build(p1, {"query_text": "shoes for running"})
    assert "for running" in reason1 or "running" in reason1
    assert "flat feet" not in reason1

    # query "women sandals" must not match tag "men" or attribute "gender": "men"
    p2 = _make_prod("P-2", tags=["men"])
    p2.attributes = {"gender": "men"}
    reason2, fields2 = reasons.build(p2, {"query_text": "women sandals"})
    assert "men" not in reason2
    assert "tags" not in fields2
