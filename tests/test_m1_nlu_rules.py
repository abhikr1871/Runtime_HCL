"""Unit tests for M1: typo-tolerant brands and Hinglish price parsing in nlu_rules.py."""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from app.pipeline import nlu_rules  # noqa: E402

KNOWN_BRANDS = ["Nike", "Adidas", "Samsung", "Puma", "Campus", "Sony", "Apple", "Peter England", "boAt", "Lenovo"]


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
        ("only puma", ["Puma"], []),
        ("not nike", [], ["Nike"]),
        ("boot cut jeans", [], []),
        ("camps shoes", [], []),
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
        "below",
        "black",
        "price",
        "show",
        "best",
        "cheap",
        "gift",
        "running",
        "water",
        "shirt",
        "boot",
        "boots",
        "camp",
        "camps",
        "cut",
        "jeans",
        "sandal",
        "sandals",
        "slipper",
        "slippers",
        "sneaker",
        "sneakers",
        "heels",
        "kurta",
        "saree",
        "hoodie",
        "jacket",
        "bag",
        "bags",
        "speaker",
        "speakers",
        "earbuds",
        "headphones",
        "charger",
        "cable",
        "mouse",
        "keyboard",
        "tablet",
        "camera",
        "fridge",
        "level",
        "sonic",
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
        ("under 5000", 5000),
        ("size 9 under 3000", 3000),
        ("32 GB laptop under 50000", 50000),
        ("5G phone under 15k", 15000),
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
        ("above Rs 2000", 2000, None),
        ("over ₹2000", 2000, None),
        ("starting from Rs 500", 500, None),
        ("min Rs 1000", 1000, None),
        ("at least Rs 2000", 2000, None),
        ("Rs 2000 se zyada", 2000, None),
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
        "iphone 15 pro",
        "phone under 32 gb",
        "laptop under 32gb ram",
        "under 5000 mah battery",
        "under 5g",
        "under 6.7 inch",
        "under 120hz",
        "max cheap shoes",
        "under doctor",
        "under week",
        "under kuch",
        "max chess",
    ],
)
def test_negative_price_cases(phrase):
    slots = nlu_rules.parse(phrase)
    assert slots.budget_max is None
    assert slots.budget_min is None


def test_complex_specs_with_valid_price():
    s1 = nlu_rules.parse("size 9 under 3000")
    assert s1.size == "9"
    assert s1.budget_max == 3000

    s2 = nlu_rules.parse("32 GB laptop under 50000")
    assert s2.budget_max == 50000

    s3 = nlu_rules.parse("5G phone under 15k")
    assert s3.budget_max == 15000


def test_catalogue_words_no_brand_false_positives():
    p = Path(__file__).resolve().parents[1] / "data_kit" / "sample_products.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            prods = json.load(f)
        non_brand_words = set()
        brand_lowers = {b.lower() for b in KNOWN_BRANDS}
        for item in prods:
            for text in [item.get("name", ""), item.get("category", ""), item.get("description", "")]:
                for word in re.findall(r"[a-zA-Z]{4,}", text.lower()):
                    if word not in brand_lowers:
                        non_brand_words.add(word)
    else:
        non_brand_words = {
            "shoes", "sneakers", "boots", "sandals", "slippers", "jeans", "pants", "shirt",
            "tshirt", "jacket", "hoodie", "watch", "camera", "phone", "laptop", "tablet",
            "speaker", "earbuds", "headphones", "bottle", "backpack", "cotton", "running",
            "casual", "formal", "sports", "battery", "screen", "cable", "charger", "wireless",
            "leather", "canvas", "travel", "office", "summer", "winter", "outdoor", "fitness",
        }

    offenders = []
    for word in sorted(non_brand_words):
        s = nlu_rules.parse(f"show me {word}", known_brands=KNOWN_BRANDS)
        if s.brands_include:
            offenders.append((word, s.brands_include))

    assert not offenders, f"Catalogue words falsely matched brands: {offenders}"
