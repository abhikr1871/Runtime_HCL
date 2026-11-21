"""Unit tests for M1: typo-tolerant brands and Hinglish price parsing in nlu_rules.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from app.pipeline import nlu_rules  # noqa: E402

KNOWN_BRANDS = ["Nike", "Adidas", "Samsung", "Puma", "Campus", "Sony", "Apple", "Peter England"]


@pytest.mark.parametrize(
    "phrase,expected_include,expected_exclude",
    [
        ("running shoes nkie", ["Nike"], []),
        ("adiddas sneakers", ["Adidas"], []),
        ("samsng phone", ["Samsung"], []),
        ("pume trackpants", ["Puma"], []),
        ("not nkie", [], ["Nike"]),
        ("nkie nahi", [], ["Nike"]),
        ("without adiddas", [], ["Adidas"]),
        ("only nkie", ["Nike"], []),
        ("only adiddas", ["Adidas"], []),
        ("nkie and adiddas", ["Nike", "Adidas"], []),
        ("formal shirt Peter England", ["Peter England"], []),
        ("not Peter England", [], ["Peter England"]),
    ],
)
def test_typo_tolerant_brands(phrase, expected_include, expected_exclude):
    slots = nlu_rules.parse(phrase, known_brands=KNOWN_BRANDS)
    assert slots.brands_include == expected_include
    assert slots.brands_exclude == expected_exclude


@pytest.mark.parametrize(
    "word",
    [
        "shoes",
        "phone",
        "under",
        "black",
        "price",
        "show",
        "best",
        "cheap",
        "gift",
        "running",
        "water",
        "shirt",
    ],
)
def test_brand_false_positive_guards(word):
    slots = nlu_rules.parse(f"show me {word}", known_brands=KNOWN_BRANDS)
    assert slots.brands_include == []
    assert slots.brands_exclude == []


@pytest.mark.parametrize(
    "phrase,expected_max",
    [
        ("3 hazaar tak", 3000),
        ("3 hazar ke andar", 3000),
        ("paanch hazaar se kam", 5000),
        ("5 hazaar tak ka", 5000),
        ("paanch sau se kam", 500),
        ("500 ke andar", 500),
        ("do sau tak", 200),
        ("under 3k", 3000),
        ("3k tak", 3000),
        ("below 2.5k", 2500),
        ("upto 3k", 3000),
        ("max 3000", 3000),
        ("3000 ke andar", 3000),
        ("3000 se kam", 3000),
        ("3000 tak", 3000),
        ("Rs 3000 budget", 3000),
        ("1 lakh tak", 100000),
        ("1.5 lakh ke andar", 150000),
        ("३ हज़ार तक", 3000),
        ("पाँच सौ से कम", 500),
    ],
)
def test_hinglish_budget_max_phrases(phrase, expected_max):
    slots = nlu_rules.parse(phrase)
    assert slots.budget_max == expected_max


@pytest.mark.parametrize(
    "phrase,expected_min,expected_max",
    [
        ("2000 se zyada", 2000, None),
        ("above 2000", 2000, None),
        ("at least 1500", 1500, None),
        ("2000 se 4000 ke beech", 2000, 4000),
        ("between 2k and 4k", 2000, 4000),
        ("between 1000 to 2500", 1000, 2500),
    ],
)
def test_budget_min_and_range_phrases(phrase, expected_min, expected_max):
    slots = nlu_rules.parse(phrase)
    assert slots.budget_min == expected_min
    assert slots.budget_max == expected_max


@pytest.mark.parametrize(
    "phrase",
    [
        "shoes size 9",
        "iPhone 15",
        "laptop with 32 GB RAM",
        "5G smartphone",
    ],
)
def test_negative_price_cases(phrase):
    slots = nlu_rules.parse(phrase)
    assert slots.budget_max is None
    assert slots.budget_min is None
