"""Unit tests for M2-B: Product Data Engineering and Indexing.

Covers:
1. Brand normalisation
2. Category normalisation
3. Size normalisation
4. Weight and unit normalisation
5. Missing fields handling (never guess)
6. Invalid prices handling
7. Duplicate and near-duplicate detection
8. Embedding generation and text preparation
9. Embedding persistence in SQLite
10. Incremental indexing and content hash caching
11. CSV ingestion via multipart upload
12. Product updates reflection
13. Product deletion / deactivation
14. Hindi, Hinglish and English product representation and tokenization
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from app.ai import embeddings  # noqa: E402
from app.schemas import Product  # noqa: E402
from app.store import db, indexer, repository  # noqa: E402
from app.store.normalise import (  # noqa: E402
    detect_duplicates,
    dimension_to_inches,
    normalize_brand,
    normalize_category,
    normalize_color,
    normalize_colors,
    normalize_product,
    normalize_size,
    normalize_sizes,
    parse_price_inr,
    volume_to_ml,
    weight_to_grams,
)


# ---------- 1. Brand Normalisation ----------
def test_brand_normalisation():
    # Canonical brands
    assert normalize_brand("boat") == "boAt"
    assert normalize_brand("BOAT") == "boAt"
    assert normalize_brand(" boAt  ") == "boAt"
    assert normalize_brand("nike") == "Nike"
    assert normalize_brand("campus") == "Campus"
    assert normalize_brand("fabindia") == "Fabindia"
    assert normalize_brand("hp") == "HP"
    assert normalize_brand("HEWLETT-PACKARD") == "HP"
    assert normalize_brand("wildcraft") == "Wildcraft"
    assert normalize_brand("quechua") == "Quechua"
    assert normalize_brand("decathlon") == "Decathlon"
    assert normalize_brand("milton") == "Milton"

    # Unknown brands: clean casing
    assert normalize_brand("super tech") == "Super Tech"
    assert normalize_brand("GENERIC BRAND") == "Generic Brand"

    # Missing brand must remain None (never guess!)
    assert normalize_brand(None) is None
    assert normalize_brand("") is None
    assert normalize_brand("   ") is None


# ---------- 2. Category Normalisation ----------
def test_category_normalisation():
    assert normalize_category("mens-shoes") == "mens-shoes"
    assert normalize_category("Men's Shoes") == "mens-shoes"
    assert normalize_category("mens shoes") == "mens-shoes"
    assert normalize_category("laptops") == "laptops"
    assert normalize_category("Laptop") == "laptops"
    assert normalize_category("backpack") == "bags"
    assert normalize_category("sports accessories") == "sports-accessories"
    assert normalize_category("mobile accessories") == "mobile-accessories"
    assert normalize_category("kitchen appliances") == "kitchen"

    # None or empty
    assert normalize_category(None) is None
    assert normalize_category("") is None


# ---------- 3. Size Normalisation ----------
def test_size_normalisation():
    # Shoe sizes
    assert normalize_size("UK 8") == "8"
    assert normalize_size("8 UK") == "8"
    assert normalize_size("UK-9") == "9"
    assert normalize_size("Size 10") == "10"
    assert normalize_size("8.5") == "8.5"

    # Clothing sizes
    assert normalize_size("s") == "S"
    assert normalize_size("small") == "S"
    assert normalize_size("M") == "M"
    assert normalize_size("medium") == "M"
    assert normalize_size("large") == "L"
    assert normalize_size("xl") == "XL"
    assert normalize_size("2xl") == "XXL"
    assert normalize_size("3XL") == "3XL"

    # Dimensions
    assert normalize_size("14 inch") == "14 inch"
    assert normalize_size("15.6\"") == "15.6 inch"

    # Multiple sizes list with deduplication
    raw_sizes = ["UK 8", "8", "UK 9", "9 UK", "size 10"]
    assert normalize_sizes(raw_sizes) == ["8", "9", "10"]


# ---------- 4. Weight and Unit Normalisation ----------
def test_weight_and_units():
    # Weights in kg, g, mg
    assert weight_to_grams("1 kg") == 1000.0
    assert weight_to_grams("1.5 kgs") == 1500.0
    assert weight_to_grams("500g") == 500.0
    assert weight_to_grams("250 gm") == 250.0
    assert weight_to_grams("250 grams") == 250.0
    assert weight_to_grams("500 mg") == 0.5

    # Ambiguous unitless weight must be None (never guess)
    assert weight_to_grams("500") is None
    assert weight_to_grams(None) is None
    assert weight_to_grams("abc") is None

    # Volumes
    assert volume_to_ml("1 L") == 1000.0
    assert volume_to_ml("750 ml") == 750.0
    assert volume_to_ml("500") is None

    # Dimensions
    assert dimension_to_inches("15.6 inch") == 15.6
    assert dimension_to_inches("14\"") == 14.0
    assert dimension_to_inches("14") is None


# ---------- Colour Normalisation ----------
def test_colour_normalisation():
    assert normalize_color("navy blue") == "blue"
    assert normalize_color("sky blue") == "blue"
    assert normalize_color("नीला") == "blue"
    assert normalize_color("neela") == "blue"
    assert normalize_color("jet black") == "black"
    assert normalize_color("काला") == "black"
    assert normalize_color("maroon") == "red"
    assert normalize_color("लाल") == "red"
    assert normalize_color("olive green") == "green"
    assert normalize_color("सफ़ेद") == "white"

    raw_colors = ["navy blue", "royal blue", "black", "काला"]
    assert normalize_colors(raw_colors) == ["blue", "black"]


# ---------- 5 & 6. Missing Fields & Invalid Prices ----------
def test_normalize_product_safeguards():
    raw = {
        "product_id": "P-TEST-1",
        "name": "Test Running Shoe",
        "category": "men's shoes",
        "brand": "nike",
        "price_inr": "₹2,499",
        "mrp_inr": "Rs. 3499",
        "stock": "-5",  # negative stock
        "weight_raw": "0.8 kg",
        "sizes": ["UK 8", "8", "9"],
        "colors": ["navy blue", "black"],
    }
    norm, fixes = normalize_product(raw, record_fixes=True)

    assert norm["brand"] == "Nike"
    assert norm["category"] == "mens-shoes"
    assert norm["price_inr"] == 2499
    assert norm["mrp_inr"] == 3499
    assert norm["stock"] == 0  # negative stock normalized to 0
    assert norm["weight_raw"] == "0.8 kg"  # raw preserved
    assert norm["weight_g"] == 800.0  # normalized g
    assert norm["sizes"] == ["8", "9"]
    assert norm["colors"] == ["blue", "black"]
    assert len(fixes) > 0

    # Never invent missing brand
    raw_nobrand = {"product_id": "P-TEST-2", "name": "Generic Item", "category": "bags"}
    norm_nobrand, _ = normalize_product(raw_nobrand)
    assert norm_nobrand.get("brand") is None

    # Invalid prices (<= 0 or unparseable)
    assert parse_price_inr("0") is None
    assert parse_price_inr("-500") is None
    assert parse_price_inr("free") is None
    assert parse_price_inr(None) is None


# ---------- 7. Duplicate and Messy Data Detection ----------
def test_duplicate_and_messy_detection():
    catalog = [
        {"product_id": "P-001", "name": "Wildcraft Trail Backpack 25L", "category": "bags", "brand": "Wildcraft", "price_inr": 2499, "stock": 10},
        {"product_id": "P-001", "name": "Wildcraft Trail Backpack 25L", "category": "bags", "brand": "Wildcraft", "price_inr": 2499, "stock": 10},  # duplicate ID
        {"product_id": "P-002", "name": "Wildcraft Trail Backpack 25L", "category": "bags", "brand": "Wildcraft", "price_inr": 2499, "stock": 10},  # duplicate title
        {"product_id": "P-003", "name": "Wildcraft Trail Backpack 25 Ltr", "category": "bags", "brand": "Wildcraft", "price_inr": 2499, "stock": 10},  # near duplicate title
        {"product_id": "P-004", "name": "", "category": "bags", "brand": "Nike", "price_inr": -100, "stock": 0},  # invalid
        {"product_id": "P-005", "name": "Cotton Kurta", "category": "mens-shirts", "brand": "Fabindia", "price_inr": 1299, "stock": 5, "description": "सूती कुर्ता", "description_lang": "hi"},
    ]
    report = detect_duplicates(catalog)

    assert "P-001" in report["exact_duplicate_ids"]
    assert "wildcraft trail backpack 25l" in report["duplicate_titles"]
    assert len(report["near_duplicate_groups"]) >= 1
    assert any(inv["product_id"] == "P-004" for inv in report["invalid_records"])
    assert "P-004" in report["out_of_stock_ids"]
    assert report["summary"]["language_breakdown"].get("hi", 0) >= 1


# ---------- 8. Embedding Generation & Text Preparation ----------
def test_embedding_text_preparation_and_serialization():
    p = Product(
        product_id="P-8001",
        name="SoundPulse Bluetooth Speaker",
        brand="boAt",
        category="mobile-accessories",
        subcategory="bluetooth speaker",
        description="Waterproof portable speaker with 12 hour battery life",
        tags=["portable", "speaker", "bluetooth", "music"],
        price_inr=1499,
        stock=20,
    )
    text = embeddings.prepare_product_text(p)
    assert "SoundPulse Bluetooth Speaker" in text
    assert "Brand: boAt" in text
    assert "Category: mobile-accessories" in text
    assert "Waterproof portable speaker" in text
    assert "Tags: portable speaker bluetooth music" in text

    # Serialization / Deserialization
    dummy_vec = np.array([0.12, -0.45, 0.99, 0.0], dtype=np.float32)
    b = embeddings.vector_to_bytes(dummy_vec)
    recovered = embeddings.bytes_to_vector(b)
    np.testing.assert_allclose(dummy_vec, recovered)


# ---------- 9. Embedding Persistence in SQLite ----------
def test_embedding_persistence(tmp_path):
    test_db = tmp_path / "embed_test.db"
    db.reset_conn()
    import os
    os.environ["DB_PATH"] = str(test_db)
    from app.config import get_settings
    get_settings.cache_clear()
    db.init_db()

    dummy_vec = np.random.randn(384).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)  # L2 normalize
    b = embeddings.vector_to_bytes(dummy_vec)

    repository.save_embedding("P-1001", "text", "test-model", b)
    loaded = repository.get_embedding("P-1001", "text")
    assert loaded is not None
    vec_recovered = embeddings.bytes_to_vector(loaded)
    np.testing.assert_allclose(dummy_vec, vec_recovered, atol=1e-6)

    all_embeds = repository.get_all_embeddings("text")
    assert "P-1001" in all_embeds

    repository.delete_embedding("P-1001", "text")
    assert repository.get_embedding("P-1001", "text") is None
    db.reset_conn()


# ---------- 10. Incremental Indexing & Content Hash ----------
def test_incremental_indexing(tmp_path):
    test_db = tmp_path / "index_test.db"
    db.reset_conn()
    import os
    os.environ["DB_PATH"] = str(test_db)
    from app.config import get_settings
    get_settings.cache_clear()
    db.init_db()

    p1 = Product(product_id="P-0010", name="Trek Backpack", category="bags", brand="Wildcraft", price_inr=2499, stock=5)
    p2 = Product(product_id="P-0020", name="Running Shoes", category="mens-shoes", brand="Nike", price_inr=3999, stock=10)
    repository.upsert_product(p1)
    repository.upsert_product(p2)

    count = indexer.rebuild_product_index()
    assert count == 2

    searchable = indexer.get_searchable_products()
    assert "P-0010" in searchable and "P-0020" in searchable

    # BM25 search
    hits = indexer.bm25_search("backpack")
    assert len(hits) >= 1
    assert hits[0][0] == "P-0010"

    # Incremental update: update P-0010 price and title
    p1_updated = Product(product_id="P-0010", name="Trek Pro Daypack", category="bags", brand="Wildcraft", price_inr=2999, stock=4)
    repository.upsert_product(p1_updated)
    indexer.update_product_index("P-0010")

    updated_searchable = indexer.get_searchable_products()
    assert updated_searchable["P-0010"].name == "Trek Pro Daypack"
    assert updated_searchable["P-0010"].price_inr == 2999

    hits_updated = indexer.bm25_search("daypack")
    assert len(hits_updated) >= 1
    assert hits_updated[0][0] == "P-0010"
    db.reset_conn()


# ---------- 11. CSV Ingestion via /ingest/csv ----------
def test_csv_ingestion(client):
    csv_content = (
        "product_id,name,category,brand,price_inr,stock,description,tags,sizes,colors,weight_raw\n"
        "P-CSV-1,Trail Running Shoes,mens-shoes,nike,3499,15,Cushioned shoes for trail running,running;trail,UK 8;9,navy blue;black,450 g\n"
        "P-CSV-2,Insulated Steel Flask,sports-accessories,Milton,799,30,Keeps water hot or cold,bottle;flask,,,0.5 kg\n"
        "P-CSV-3,Broken Item,,Unknown,-100,5,No category and bad price,,,,,\n"
    )
    file_bytes = csv_content.encode("utf-8")
    resp = client.post("/ingest/csv", files={"file": ("products.csv", io.BytesIO(file_bytes), "text/csv")})
    assert resp.status_code == 200
    data = resp.json()

    assert data["created"] == 2
    assert len(data["errors"]) == 1
    assert data["errors"][0]["product_id"] == "P-CSV-3"

    # Verify searchability
    r1 = client.get("/products/P-CSV-1")
    assert r1.status_code == 200
    p1 = r1.json()
    assert p1["brand"] == "Nike"
    assert p1["weight_g"] == 450.0

    r2 = client.get("/products/P-CSV-2")
    assert r2.status_code == 200
    p2 = r2.json()
    assert p2["weight_g"] == 500.0


# ---------- 12 & 13. Product Updates, Deactivation, and Stock Changes ----------
def test_product_deactivation_and_stock(client):
    new_prod = {
        "product_id": "P-STOCK-1",
        "name": "Limited Edition Watch",
        "category": "watches",
        "brand": "Titan",
        "price_inr": 4999,
        "stock": 3,
        "tags": ["watch", "titan"],
    }
    r = client.post("/ingest", json=[new_prod])
    assert r.status_code == 200 and r.json()["created"] == 1

    # Product is searchable
    assert "P-STOCK-1" in indexer.get_searchable_products()

    # Soft delete / deactivate
    repository.deactivate_product("P-STOCK-1")
    indexer.update_product_index("P-STOCK-1")

    # Inactive product must be excluded from searchable products
    assert "P-STOCK-1" not in indexer.get_searchable_products()
    bm25_hits = indexer.bm25_search("titan watch")
    assert all(pid != "P-STOCK-1" for pid, _ in bm25_hits)


# ---------- 14. Hindi, Hinglish and English Tokenization & BM25 ----------
def test_multilingual_representation_and_bm25():
    hindi_doc = "fabindia ka सूती कुर्ता रोज़ाना पहनने के लिए आरामदायक"
    tokens = indexer.tokenize(hindi_doc)
    assert "सूती" in tokens
    assert "कुर्ता" in tokens
    assert "fabindia" in tokens
    assert "आरामदायक" in tokens
    # Stopwords like 'ka', 'ke', 'liye' filtered out
    assert "ka" not in tokens
    assert "liye" not in tokens

    p_hi = Product(
        product_id="P-HI-01",
        name="Cotton Kurta",
        brand="Fabindia",
        category="mens-shirts",
        description="सूती कुर्ता, रोज़ पहनने के लिए आरामदायक",
        description_lang="hi",
        price_inr=1290,
        stock=10,
    )
    doc_text = embeddings.prepare_product_text(p_hi)
    assert "सूती कुर्ता" in doc_text
    assert "Brand: Fabindia" in doc_text
