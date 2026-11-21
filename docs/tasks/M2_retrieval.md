# M2: Retrieval (search, ranking, ingest, normaliser)

**Goal:** the top 3 results genuinely fit the request, constraints are never broken, and new products are searchable instantly.

**You own:** `api/app/pipeline/filters.py`, `retrieval.py`, `rerank.py`, `diversity.py`, `api/app/store/indexer.py`, `normalise.py`, `api/app/ai/embeddings.py`, `api/app/routes/ingest.py`

## Tasks

**0:50-2:00**
- [ ] `indexer.py` + `retrieval.py`: BM25 (`rank_bm25`) over name, brand, tags, description
- [ ] `embeddings.py`: e5 embeddings (`passage:` / `query:` prefixes), stored in the `embeddings` table, loaded into NumPy
- [ ] `retrieval.rank`: BM25 ranking + embedding ranking fused with the existing `rrf_fuse()`
- [ ] Add `RETRIEVAL_MODE=bm25|dense|hybrid` so M5 can run the ablation

**2:00-3:30**
- [ ] `filters.py`: size and colour filters via `json_each(sizes_json / colors_json)`
- [ ] `normalise.py`: sizes, colour synonyms; a report of every fix for the data note
- [ ] `ingest.py`: re-embed only changed rows; accept CSV upload
- [ ] Check a Hindi query finds English products

**3:30-4:30 (stretch)**
- [ ] Cross-encoder reranker on the top 20 (`ENABLE_RERANKER`); keep it only if the ablation shows a gain
- [ ] MMR diversity over embeddings

## Done when
- `tests/test_api.py::test_budget_search_respects_constraint` and `test_ingest_then_searchable` pass with real data
- Add `tests/test_filters.py`: no result ever breaks budget, brand or stock
- Ablation numbers handed to M5

## Be ready to explain
Why filter before ranking; BM25 vs embeddings and why fuse them; how RRF works; why no vector database at this size.
