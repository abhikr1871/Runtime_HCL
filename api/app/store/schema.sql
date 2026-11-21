-- SQLite schema. Owner: M3. Raw and normalised values are stored side by side.
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,           -- P-0001
    source TEXT NOT NULL DEFAULT 'ingest', -- dummyjson | generated | ingest | sample
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT,
    brand TEXT,                            -- NULL = not listed (never guessed)
    price_inr INTEGER,
    mrp_inr INTEGER,
    stock INTEGER NOT NULL DEFAULT 0,
    rating REAL,
    rating_count INTEGER,
    description TEXT NOT NULL DEFAULT '',
    description_lang TEXT NOT NULL DEFAULT 'en',
    attributes_json TEXT NOT NULL DEFAULT '{}',
    sizes_json TEXT NOT NULL DEFAULT '[]',
    colors_json TEXT NOT NULL DEFAULT '[]',
    tags_json TEXT NOT NULL DEFAULT '[]',
    weight_raw TEXT,
    weight_g REAL,
    image_path TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    version INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_products_cat ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_price ON products(price_inr);

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(product_id),
    rating REAL,
    text TEXT NOT NULL,
    lang TEXT NOT NULL DEFAULT 'en',
    author_alias TEXT,
    injection_flag INTEGER NOT NULL DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS shoppers (
    customer_id TEXT PRIMARY KEY,          -- C-001
    display_name TEXT,
    city TEXT,
    preferred_language TEXT DEFAULT 'en',
    budget_min_inr INTEGER,
    budget_max_inr INTEGER,
    fav_brands_json TEXT DEFAULT '[]',
    sizes_json TEXT DEFAULT '{}',
    interests_json TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES shoppers(customer_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    qty INTEGER NOT NULL DEFAULT 1,
    price_paid_inr INTEGER,
    ordered_at TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT,
    customer_id TEXT,
    endpoint TEXT,
    record_json TEXT NOT NULL,             -- full redacted record (see observe/trace_store.py)
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS embeddings (
    product_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'text',
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (product_id, kind)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    ingest_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL,
    action TEXT NOT NULL,                  -- create | update
    fields_changed TEXT,
    at TEXT NOT NULL DEFAULT (datetime('now'))
);
