"""M1 (understanding + answer) unit tests: Hindi/Hinglish, typos, intents, clarify policy, reasons, validator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from app.pipeline import clarify, nlu_rules, reasons, validator  # noqa: E402
from app.schemas import Product, SessionState  # noqa: E402

BRANDS = ["Nike", "boAt", "Fabindia", "Campus", "JBL"]
VOCAB = {"bluetooth", "speaker", "speakers", "running", "shoes", "earbuds", "wireless", "kurta", "trekking", "backpack",
         "pressure", "cooker", "smartphones", "watches", "laptops", "bottle"}


def P(pid="P-0001", price=1000, stock=10, brand="Nike", category="audio"):
    return Product(product_id=pid, name=f"Item {pid}", category=category, brand=brand, price_inr=price, stock=stock, rating=4.2)


# ---------- Hindi / Hinglish ----------
def test_hindi_product_words_become_english_search_words():
    s = nlu_rules.parse("प्रेशर कुकर चाहिए", BRANDS, VOCAB)
    assert s.language == "hi" and "pressure cooker" in s.query_text
    assert "kurta" in nlu_rules.parse("कुर्ता दिखाओ", BRANDS, VOCAB).query_text


def test_hindi_and_hinglish_budgets():
    s = nlu_rules.parse("मुझे 2000 रुपये से कम में ईयरबड्स चाहिए", BRANDS, VOCAB)
    assert s.budget_max == 2000 and "earbuds" in s.query_text and s.language == "hi"
    assert nlu_rules.parse("speaker 3 hazaar tak", BRANDS, VOCAB).budget_max == 3000
    assert nlu_rules.parse("जूते ३००० तक", BRANDS, VOCAB).budget_max == 3000      # Devanagari digits


def test_number_words_only_count_as_whole_words():
    assert nlu_rules.parse("dove soap under 500", BRANDS, VOCAB).budget_max == 500
    assert nlu_rules.parse("show me dove soap", BRANDS, VOCAB).budget_max is None   # "do" inside "dove" is not 2
    assert nlu_rules.parse("do hazaar tak ka speaker", BRANDS, VOCAB).budget_max == 2000


def test_hinglish_word_mapping():
    assert "shoes" in nlu_rules.parse("mujhe joote chahiye", BRANDS, VOCAB).query_text


# ---------- typos ----------
def test_typos_corrected_against_catalogue_words_and_brands():
    s = nlu_rules.parse("blutooth speaker", BRANDS, VOCAB)
    assert "bluetooth" in s.query_text and s.corrections.get("blutooth") == "bluetooth"
    s = nlu_rules.parse("nkie running shoes", BRANDS, VOCAB)
    assert s.brands_include == ["Nike"]


def test_known_words_are_never_corrected():
    s = nlu_rules.parse("cheap gift for my father", BRANDS, VOCAB)
    assert s.corrections == {}


# ---------- intents ----------
def test_brand_only_message_is_a_refinement_but_product_plus_brand_is_a_search():
    assert nlu_rules.parse("only Nike", BRANDS, VOCAB).intent == "refine"
    assert nlu_rules.parse("not boAt", BRANDS, VOCAB).intent == "refine"
    s = nlu_rules.parse("running shoes not Nike", BRANDS, VOCAB)
    assert s.intent == "search" and s.brands_exclude == ["Nike"]
    assert nlu_rules.parse("kurta without Fabindia", BRANDS, VOCAB).brands_exclude == ["Fabindia"]


def test_cheaper_follow_up_stays_a_refinement():
    s = nlu_rules.parse("show me cheaper ones", BRANDS, VOCAB)
    assert s.intent == "refine" and s.wants_cheaper


def test_out_of_scope_and_not_sold():
    assert nlu_rules.parse("I want a life insurance plan", BRANDS, VOCAB).intent == "out_of_scope"
    assert nlu_rules.parse("Book me flight tickets to Goa", BRANDS, VOCAB).intent == "out_of_scope"
    assert nlu_rules.parse("Who will win the cricket match today?", BRANDS, VOCAB).intent == "out_of_scope"
    assert nlu_rules.parse("Do you sell electric cars?", BRANDS, VOCAB).intent == "not_sold"
    assert nlu_rules.parse("cricket bat", BRANDS, VOCAB).intent == "search"      # a product, not out of scope
    assert nlu_rules.parse("running shoes for flat feet", BRANDS, VOCAB).intent == "search"   # "flat" is not real estate
    assert nlu_rules.parse("house slippers", BRANDS, VOCAB).intent == "search"
    assert nlu_rules.parse("2bhk flat for rent", BRANDS, VOCAB).intent == "not_sold"


def test_search_synonyms():
    assert "smartphones" in nlu_rules.expand_for_search("phone")
    assert "watches" in nlu_rules.expand_for_search("watch")


# ---------- clarify policy ----------
def _state(query, **filters):
    return SessionState(session_id="s", customer_id="C-001", filters={"query_text": query, **filters})


def test_clarify_only_for_broad_requests():
    many = [P(f"P-{i:04d}") for i in range(12)]
    few = many[:3]
    assert clarify.next_question(_state("I need a laptop"), many[:5], "en")       # broad item
    assert clarify.next_question(_state("I need a laptop"), few, "en") is None    # 3 or fewer: just show them
    assert clarify.next_question(_state("need earbuds"), many, "en")              # one word, many matches
    assert clarify.next_question(_state("wireless earbuds"), many, "en") is None  # specific enough
    assert clarify.next_question(_state("earbuds", brands_exclude=["boAt"]), many, "en") is None
    assert clarify.next_question(_state("laptop", budget_max=50000), many, "en") is None
    assert clarify.next_question(_state("kurta"), few, "en") is None              # narrow and few matches
    spread = [P("P-0001", price=38990), P("P-0002", price=72990)]                 # few, but prices far apart
    assert clarify.next_question(_state("I need a laptop"), spread, "en")[0] == "budget_max"


def test_clarify_asks_in_the_shoppers_language_and_only_once():
    same = [P(f"P-{i:04d}") for i in range(5)]       # nothing varies -> budget question
    q = clarify.next_question(_state("mujhe ek phone chahiye"), same, "hinglish")
    assert q and "budget" in q[1].lower()
    st = _state("phone")
    st.asked_slots = ["budget_max"]
    assert clarify.next_question(st, same, "en") is None


# ---------- reasons ----------
def test_reasons_in_three_languages_keep_numbers():
    p = P(price=2099, stock=2)
    en = reasons.build(p, {"budget_max": 2500}, "en")[0]
    hi = reasons.build(p, {"budget_max": 2500}, "hi")[0]
    hg = reasons.build(p, {"budget_max": 2500}, "hinglish")[0]
    for r in (en, hi, hg):
        assert "₹2,099" in r and "₹2,500" in r and "2" in r
    assert "बजट" in hi and "budget ke andar" in hg


def test_over_budget_item_never_claims_to_be_under_budget():
    assert "under your" not in reasons.build(P(price=3100), {"budget_max": 2500}, "en")[0]


# ---------- validator ----------
def test_validator_checks_stock_claims():
    p = P(price=999, stock=2)
    assert validator.check("Only 2 left of P-0001 at ₹999", [p]).ok
    assert not validator.check("Only 5 left of P-0001", [p]).ok
    assert validator.check("सिर्फ़ 2 बचे हैं", [p]).ok
    assert not validator.check("P-0001 is out of stock", [p]).ok
