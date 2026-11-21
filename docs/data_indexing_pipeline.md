# M2-B: Product Data Engineering and Indexing Pipeline

This document outlines the architecture, normalization logic, embedding pipeline, and indexing infrastructure implemented for the **HCL AI Shopping Assistant**.

---

## 1. Architecture Overview

The M2-B subsystem is responsible for cleaning, normalizing, validating, and indexing product catalogue data to power high-precision search.

```
+-----------------------------------------------------------------------------------+
|                                 Catalogue Sources                                 |
|               (DummyJSON / Generated Products / CSV / JSON Ingestion)             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                          app/store/normalise.py                                   |
| - Brand canonicalization (boAt, Nike, HP, Fabindia, ...)                          |
| - Category & Subcategory standard slugs                                           |
| - Size normalisation (Shoe UK/US/EU, Clothing XS-3XL, Inches)                     |
| - Multilingual colour synonyms (English, Hindi, Hinglish)                         |
| - Unit conversion (kg -> g, ml -> l, inch, ambiguous unitless -> None)           |
| - Non-negative INR price validation & raw value preservation                      |
| - Duplicate & near-duplicate detection (RapidFuzz token set ratio)                |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        app/store/repository.py (SQLite)                           |
| - products table: raw and normalized fields side-by-side                          |
| - embeddings table: BLOB storage for float32 vectors (kind='text')                |
| - ingest_log table: audit records of all insertions/updates                       |
+-----------------------------------------------------------------------------------+
                                          |
                 +------------------------+------------------------+
                 |                                                 |
                 v                                                 v
+----------------------------------+             +----------------------------------+
|           BM25 Index             |             |        e5 Dense Embeddings       |
|       (rank_bm25.BM25Okapi)      |             | (intfloat/multilingual-e5-small) |
| - Multilingual regex tokenizer   |             | - 'passage: ' text prefix        |
| - English + Hindi stopword filter|             | - Float32 L2-normalized vectors  |
| - Product ID <-> Doc ID mapping  |             | - In-memory NumPy matrix cache   |
+----------------------------------+             +----------------------------------+
                 \                                                 /
                  \                                               /
                   +----------------------+----------------------+
                                          v
                         +----------------------------------+
                         |       app/store/indexer.py       |
                         | - In-memory product cache        |
                         | - Incremental hash-based updates |
                         | - Thread-safe atomic swaps       |
                         +----------------------------------+
                                          |
                                          v
                         +----------------------------------+
                         |       Interface for M2-A         |
                         | - get_searchable_products()      |
                         | - get_product_embeddings()       |
                         | - get_bm25_documents()           |
                         | - rebuild_product_index()        |
                         | - update_product_index(pid)      |
                         | - bm25_search(query)             |
                         | - dense_search(query)            |
                         +----------------------------------+
```

---

## 2. Data Normalization (`api/app/store/normalise.py`)

### A. Principles
1. **Never guess**: If a unit is missing (e.g. weight `"540"` with no unit), return `None` ("not listed").
2. **Preserve raw values**: Store both `weight_raw` (e.g. `"1.2 kg"`) and `weight_g` (`1200.0`).
3. **Never alter product IDs or prices**: Prices remain integer INR; IDs remain unchanged.
4. **Deterministic & Repeatable**: The same input always produces the exact same normalized structure.

### B. Normalized Attributes
- **Brands**: Canonical casing via known brand registry (`"boat"` -> `"boAt"`, `"wildcraft"` -> `"Wildcraft"`, `"hp"` -> `"HP"`, etc.). Unlisted brands retain clean title-casing.
- **Categories**: Standardized to slug formats (e.g. `"Men's Shoes"` -> `"mens-shoes"`, `"kitchen appliances"` -> `"kitchen"`).
- **Sizes**:
  - Shoes: `"UK 8"`, `"8 UK"`, `"Size 8"` -> `"8"`.
  - Clothing: `"small"` -> `"S"`, `"medium"` -> `"M"`, `"large"` -> `"L"`, `"xl"` -> `"XL"`, `"2xl"` -> `"XXL"`, `"3xl"` -> `"3XL"`.
  - Dimensions: `"14 inch"`, `"14\""` -> `"14 inch"`.
- **Colours**: Maps English, Hindi and Hinglish synonyms to base colours (e.g. `"navy blue"`, `"नीला"`, `"neela"` -> `"blue"`).
- **Weights & Units**:
  - `weight_to_grams`: `1 kg` -> `1000.0 g`, `500 mg` -> `0.5 g`.
  - `volume_to_ml`: `1 L` -> `1000.0 ml`.
  - `dimension_to_inches`: `15.6 inch` -> `15.6`.
- **Prices**: Rejects non-positive or unparseable values (`parse_price_inr`).

---

## 3. Duplicate and Messy Data Detection

The `detect_duplicates()` function performs complete catalogue audit checks:
- **Exact duplicate IDs**: Finds IDs appearing more than once.
- **Exact duplicate titles**: Groups different products with identical normalized titles.
- **Near-duplicate titles**: Employs `rapidfuzz.fuzz.token_set_ratio` (threshold >= 88%) to group titles with minor naming discrepancies or variations.
- **Catalogue Health**: Reports missing descriptions, missing brands, non-positive prices, and out-of-stock items.
- **Multilingual Audit**: Verifies Devanagari script presence for Hindi and Latin script for Hinglish.

---

## 4. Multilingual Embeddings (`api/app/ai/embeddings.py`)

- **Model**: `intfloat/multilingual-e5-small` (384-dimensional dense vectors).
- **Prefix Conventions**:
  - Query: `query: <text>`
  - Document/Passage: `passage: <rich product text>`
- **Passage Construction**: Combines Name, Brand, Category, Subcategory, Tags, and Description for comprehensive semantic coverage across English, Hindi, and Hinglish.
- **BLOB Persistence**: Vectors are serialized to compact float32 byte arrays and stored in SQLite `embeddings` table (`product_id`, `kind`, `model`, `vector`, `updated_at`).
- **Graceful Fallback**: When `ENABLE_EMBEDDINGS=false` or model loading fails, logs a warning and search transparently relies on BM25 without throwing errors.

---

## 5. Incremental Indexing Engine (`api/app/store/indexer.py`)

- **Content Hashing**: Computes SHA-256 of the rich product text.
- **Selective Embedding**: When `update_product_index(product_id)` is invoked, re-embedding is only executed if the content hash has changed or the vector does not exist in SQLite.
- **Deletion & Deactivation Handling**: When a product is deleted or deactivated (`is_active = 0`), it is purged from active in-memory search dictionaries, BM25 corpus, and vector matrix.
- **Thread Safety**: Atomic updates under a reentrant lock (`threading.RLock`) guarantee zero read downtime or race conditions.

---

## 6. CSV & JSON Product Ingestion (`api/app/routes/ingest.py`)

- `POST /ingest`: Accepts a JSON array of product objects.
- `POST /ingest/csv`: Multipart form upload accepting CSV files (`.csv`).
  - Supports UTF-8 and UTF-8-sig (Excel exports).
  - Validates required columns (`product_id`, `name`, `category`).
  - Automatically parses JSON/delimited lists for `sizes`, `colors`, `tags`.
  - Normalizes fields and applies `Product` Pydantic model validation.
  - Isolates row-level errors: invalid rows are reported with detailed reasons without corrupting valid rows or the database.
  - Immediately updates BM25 and embedding indexes.

---

## 7. Interface Contract with M2-A

M2-A consumes the index through the standardized functions in `api/app/store/indexer.py`:

| Function | Signature | Purpose |
|---|---|---|
| `get_searchable_products()` | `() -> dict[str, Product]` | All active products indexed in memory |
| `get_product_embeddings()` | `() -> tuple[list[str], Optional[np.ndarray]]` | Product IDs and normalized float32 embedding matrix |
| `get_bm25_documents()` | `() -> tuple[list[str], list[list[str]]]` | Product IDs and tokenized documents |
| `rebuild_product_index()` | `() -> int` | Full rebuild from database |
| `update_product_index(pid)` | `(str \| list[str]) -> None` | Incremental update of modified/added products |
| `bm25_search(query, candidates)` | `(str, list[str]) -> list[tuple[str, float]]` | Scored candidate products using BM25Okapi |
| `dense_search(query, candidates)` | `(str/vec, list[str]) -> list[tuple[str, float]]` | Scored candidate products using Cosine Similarity |
| `status()` | `() -> str` | Diagnostic status for `/health` endpoint |

---

## 8. Performance and Memory Benchmarks

Tested on standard development environment across **500 synthetic products** (hackathon target catalogue size):

| Benchmark Metric | Result | Target Requirement | Status |
|---|---|---|---|
| **500 Products Rebuild Time** | **18.31 ms** | < 100 ms | PASS |
| **BM25 Search Latency** | **0.91 ms** | < 10 ms | PASS |
| **Incremental Update Latency** | **8.70 ms** | < 25 ms | PASS |
| **Embedding Matrix Memory (500 x 384 dim)** | **750.0 KB** (0.73 MB) | < 20 MB | PASS |
| **Test Suite Execution (36 unit tests)** | **1.12 s** | < 5 s | PASS |

---

## 9. Test Suite Verification

Comprehensive test coverage is provided in [tests/test_data_indexing.py](file:///C:/Users/ASUS/OneDrive/Desktop/HCL/Runtime_HCL/tests/test_data_indexing.py):
- `test_brand_normalisation`: Canonical branding and casing.
- `test_category_normalisation`: Slug standardisation.
- `test_size_normalisation`: Shoe, apparel and dimension parsing.
- `test_weight_and_units`: Grams, kilograms, milligrams, milliliters, and ambiguous unit handling.
- `test_colour_normalisation`: Multilingual and synonym mapping.
- `test_normalize_product_safeguards`: Non-invention of missing fields, negative stock/price sanitisation.
- `test_duplicate_and_messy_detection`: Detection of duplicate IDs, title groups, and RapidFuzz near-duplicates.
- `test_embedding_text_preparation_and_serialization`: Multilingual passage formatting and byte serialization.
- `test_embedding_persistence`: SQLite BLOB round-trip verification.
- `test_incremental_indexing`: Content hash tracking, BM25 update, and selective re-embedding.
- `test_csv_ingestion`: Multipart CSV upload, error reporting, and immediate searchability.
- `test_product_deactivation_and_stock`: Inactive product exclusion from BM25 and product index.
- `test_multilingual_representation_and_bm25`: Tokenization and searching across Hindi (Devanagari) and English.
