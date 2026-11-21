# Runtime_HCL: AI Shopping Assistant

A grounded shopping assistant for English, Hindi and Hinglish, built for the HCLTech AI Engineer Hackathon.
**Code filters, ranks and fact-checks every product; the LLM only understands the request and words the reply**,
so over-budget, out-of-stock or invented products cannot reach the shopper.

- Design (5 pages): [`docs/Design_5_Pages.pdf`](docs/Design_5_Pages.pdf)
- Architecture (1 page): [`docs/Architecture.pdf`](docs/Architecture.pdf)
- Search & Retrieval Engine: [`docs/retrieval.md`](docs/retrieval.md)
- Task sheets per member: [`docs/tasks/`](docs/tasks/README.md)

## Quick start (local, about 5 minutes)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                    # MOCK_LLM=true works without Ollama

# terminal 1: API (from the api/ folder)
cd api && uvicorn app.main:app --reload --port 8000

# terminal 2: UI (from the repo root)
streamlit run ui/app.py

# tests (from the repo root)
pytest -q
```

The API seeds 12 sample products on first start, so everything works before the real catalogue is built.
To build the real catalogue: `python data_kit/fetch_dummyjson.py`, then the other `data_kit/` scripts, then `python data_kit/build_db.py`.

**With the real LLM:** install [Ollama](https://ollama.com), run `ollama pull qwen2.5:7b-instruct`, set `MOCK_LLM=false` in `.env`.

## Docker

```bash
cp .env.example .env
docker compose up --build      # API on :8000, UI on :8501; Ollama runs on the host
```

## API (contract in `api/app/schemas.py`)

| Endpoint | Purpose |
| --- | --- |
| `POST /chat` | `{session_id, customer_id, message}` → answer, products with reasons, sources, follow-up question |
| `GET /products/{id}` | One product with reviews, to check any claim |
| `POST /ingest` | Add or update products; searchable on the next request |
| `GET /trace/{trace_id}` | Full record of how a reply was made |
| `GET /health` | Readiness |
| `GET /sessions/{id}?customer_id=` | Remembered filters (UI chips) |

```bash
curl -s localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id":"demo","customer_id":"C-001","message":"gift for my father who likes trekking under 3000"}'
curl -s localhost:8000/trace/<trace_id>
curl -s localhost:8000/products/P-9001
```

## How it works

1. **Understand:** guard (privacy, injection, PII) → rules (price, brand, size, negation, references) → LLM fallback only if unclear → session state.
2. **Retrieve:** SQL hard filters (budget, brand, stock) → keyword/BM25 + multilingual embeddings → Reciprocal Rank Fusion → diversity.
3. **Answer:** reasons built from database fields → LLM words the reply → validator checks every ₹ amount and product ID → trace saved.

## Repository layout

| Folder | What | Owner |
| --- | --- | --- |
| `api/app/pipeline/` | The request pipeline, one module per step | M1, M2, M3 |
| `api/app/store/` | SQLite schema, repository, normaliser, index | M2, M3 |
| `api/app/ai/` | LLM client, embeddings, prompts | M1, M2 |
| `api/app/routes/` | FastAPI endpoints | M3 |
| `ui/` | Streamlit app and pages | M4 |
| `data_kit/` | Data download, generation, validation, DB build | M5 |
| `eval/` | Labelled cases, harness, ablation, report | M5 |
| `tests/` | pytest suite (runs with the mock LLM) | everyone |

## Assumptions and limitations

- Identity comes from `customer_id` in the request; there are no passwords (hackathon scope).
- Ollama runs on the host; any hosted LLM is used only if organisers approve and is disclosed here.
- Catalogue: DummyJSON demo data (MIT-licensed project) plus generated products; no real customer data.

## Submission checklist

- [ ] Tag `final` · [ ] `docker compose up` from a fresh clone · [ ] data note and prompts · [ ] evaluation report
- [ ] 3 sample audit records in `docs/audit_samples/` · [ ] `docs/ai_usage.md` · [ ] `docs/contributions.md` · [ ] signed declarations
