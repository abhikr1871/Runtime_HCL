"""Unit and integration tests for M2 search and retrieval engine.

Tests all requirements:
1. Brand filtering
2. Budget filtering
3. Stock filtering
4. Category filtering
5. Size and colour filtering
6. Excluded brands
7. BM25 retrieval
8. Dense retrieval
9. Hybrid retrieval (RRF)
10. Hindi and Hinglish queries
11. Empty results
12. Duplicate results and diversity
13. Retrieval mode switching
14. Interface contract (search_products)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

import pytest

from app.pipeline import diversity, filters, retrieval
from app.schemas import Product
from app.store import db, indexer, repository


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    """Fresh, isolated DB with the 12 sample products for every test.

    Uses a temporary DB_PATH so the tests never read the real data/shop.db (265 products)
    and give the same result on every laptop.
    """
    from app.config import get_settings

    monkeypatch.setenv("DB_PATH", str(tmp_path / "retrieval_test.db"))
    monkeypatch.setenv("ENABLE_EMBEDDINGS", "false")
    get_settings.cache_clear()
    db.reset_conn()
    db.init_db()
    db.seed_sample_if_empty()
    indexer.rebuild()
    yield
    db.reset_conn()
    get_settings.cache_clear()


# 1. Brand filtering
def test_brand_filtering():
    # Only Nike
    prods = filters.apply({"brands_include": ["Nike"]})
    assert len(prods) > 0
    assert all(p.brand and p.brand.lower() == "nike" for p in prods)

    # Multiple brands
    prods_multi = filters.apply({"brands_include": ["Nike", "Campus"]})
    assert len(prods_multi) == 2
    assert {p.brand.lower() for p in prods_multi} == {"nike", "campus"}

    # Non-existent brand
    assert len(filters.apply({"brands_include": ["NonExistentBrand"]})) == 0


# 2. Budget filtering
def test_budget_filtering():
    # Under 3000
    under_3000 = filters.apply({"budget_max": 3000})
    assert len(under_3000) > 0
    assert all(p.price_inr is not None and p.price_inr <= 3000 for p in under_3000)

    # Above 30000 (laptops)
    above_30k = filters.apply({"budget_min": 30000})
    assert len(above_30k) > 0
    assert all(p.price_inr is not None and p.price_inr >= 30000 for p in above_30k)

    # Between 1000 and 2000
    between = filters.apply({"budget_min": 1000, "budget_max": 2000})
    assert len(between) > 0
    assert all(1000 <= p.price_inr <= 2000 for p in between)


# 3. Stock filtering
def test_stock_filtering():
    # Default in-stock only: P-9010 (stock=0) must never be returned
    in_stock = filters.apply({})
    assert len(in_stock) > 0
    assert all(p.stock > 0 for p in in_stock)
    assert not any(p.product_id == "P-9010" for p in in_stock)

    # Out of stock matches
    oos = filters.out_of_stock_matches({})
    assert any(p.product_id == "P-9010" for p in oos)
    assert all(p.stock <= 0 for p in oos)


# 4. Category filtering
def test_category_filtering():
    laptops = filters.apply({"category": "laptops"})
    assert len(laptops) == 2
    assert all(p.category == "laptops" for p in laptops)

    # Case-insensitive
    shoes = filters.apply({"category": "MENS-SHOES"})
    assert len(shoes) >= 2
    assert all(p.category == "mens-shoes" for p in shoes)


# 5. Size and colour filtering
def test_size_and_colour_filtering():
    # Size 8
    size_8 = filters.apply({"size": "8"})
    assert len(size_8) >= 2
    assert all("8" in p.sizes or any("8" in str(sz) for sz in p.sizes) for p in size_8)

    # Size M
    size_m = filters.apply({"size": "M"})
    assert any(p.product_id == "P-9011" for p in size_m)


# 6. Excluded brands
def test_excluded_brands():
    all_speakers = filters.apply({"category": "mobile-accessories"})
    no_boat = filters.apply({"category": "mobile-accessories", "brands_exclude": ["boAt"]})
    assert len(no_boat) < len(all_speakers)
    assert not any(p.brand and p.brand.lower() == "boat" for p in no_boat)


# 7. BM25 retrieval
def test_bm25_retrieval():
    all_prods = repository.find_products()
    results = retrieval.bm25_rank("bluetooth speaker", all_prods)
    assert len(results) > 0
    top_p, score = results[0]
    assert score > 0
    assert "speaker" in top_p.name.lower() or "speaker" in top_p.tags

    # Typo tolerance
    typo_results = retrieval.bm25_rank("blutooth speeker", all_prods)
    assert len(typo_results) > 0
    assert "speaker" in typo_results[0][0].name.lower()


# 8. Dense retrieval
def test_dense_retrieval():
    all_prods = repository.find_products()
    # When embeddings are disabled, dense_rank returns empty list or fallback cleanly
    results = retrieval.dense_rank("waterproof audio for party", all_prods)
    assert isinstance(results, list)


# 9. Hybrid retrieval (RRF)
def test_hybrid_retrieval():
    all_prods = repository.find_products()
    results = retrieval.hybrid_rank("trekking backpack", all_prods)
    assert len(results) > 0
    top_p, score = results[0]
    assert "trekking" in top_p.tags or "backpack" in top_p.tags or "trail" in top_p.name.lower()


# 10. Hindi and Hinglish queries
def test_hindi_hinglish_queries():
    all_prods = repository.find_products()
    # Hindi query
    hi_results = retrieval.bm25_rank("कुर्ता", all_prods)
    assert any(p.product_id == "P-9011" for p, _ in hi_results)

    # Hinglish query
    hinglish_results = retrieval.bm25_rank("mujhe running shoes chahiye", all_prods)
    assert any("running" in p.name.lower() for p, _ in hinglish_results[:3])


# 11. Empty results handling
def test_empty_results():
    all_prods = repository.find_products()
    # Completely unmatched content query
    assert len(retrieval.bm25_rank("aeroplane helicopter rocket", all_prods)) == 0

    # Over-constrained filter
    impossible = filters.apply({"budget_max": 50})
    assert len(impossible) == 0


# 12. Duplicate results and diversity
def test_diversity_and_duplicates():
    all_prods = repository.find_products()
    # If top candidates contain multiple items of the same brand, diversity picks distinct brands first
    ranked = retrieval.rank("laptop", all_prods)
    diversified = diversity.diversify(ranked, k=2)
    assert len(diversified) == 2
    # Lenovo vs ASUS: brands must be distinct
    brands = [p.brand for p in diversified if p.brand]
    assert len(brands) == len(set(brands))


# 13. Retrieval mode switching
def test_retrieval_mode_switching():
    all_prods = repository.find_products()
    r_bm25 = retrieval.rank("running shoes", all_prods, mode="bm25")
    assert len(r_bm25) > 0

    r_dense = retrieval.rank("running shoes", all_prods, mode="dense")
    assert len(r_dense) > 0

    r_hybrid = retrieval.rank("running shoes", all_prods, mode="hybrid")
    assert len(r_hybrid) > 0


# 14. Interface contract (search_products)
def test_search_products_interface_contract():
    res = retrieval.search_products(
        filters={"category": "mens-shoes", "budget_max": 3000},
        query="running shoes",
        top_k=5,
    )
    assert isinstance(res, list)
    assert len(res) > 0
    item = res[0]
    # Check all required keys in the contract
    assert "product_id" in item
    assert "name" in item
    assert "brand" in item
    assert "price" in item
    assert "in_stock" in item
    assert "score" in item
    assert "retrieval_method" in item
    # Ensure hard constraints were respected
    assert all(it["price"] <= 3000 for it in res)
    assert all(it["in_stock"] is True for it in res)
