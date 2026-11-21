# Team task sheets

| Member | Role | Sheet |
| --- | --- | --- |
| M1 (lead) | AI lead: understanding, answer, validator, integration | [M1_ai_lead.md](M1_ai_lead.md) |
| M2 | Retrieval: search, ranking, ingest, normaliser | [M2_retrieval.md](M2_retrieval.md) |
| M3 | Backend: API, database, session state, privacy, traces | [M3_backend.md](M3_backend.md) |
| M4 | Frontend and packaging: Streamlit UI, Docker, README | [M4_frontend.md](M4_frontend.md) |
| M5 | Data and evaluation: data kit, test set, report | [M5_data_eval.md](M5_data_eval.md) |

## Rules for everyone

1. **Contracts are frozen** (`api/app/schemas.py`). Need a change? Ask the lead first; everyone depends on it.
2. **Work on your own branch**: `git checkout -b m2-retrieval`; open a pull request into `main`. Pull `main` every hour.
3. **Only edit files you own.** If you must touch someone else's file, tell them first.
4. **Run `pytest -q` before every push.** Add a test for anything you build.
5. **Commit every hour** (the judges look at Git history across members).
6. **Gate at 2:00:** one question works end to end with the real data. **Feature freeze at 4:30:** bug fixes only.
7. **Everyone must explain the whole system** in Q&A, not just their part. Read the 5-page design and the architecture PDF.

## Setup (first 30 minutes, every laptop)

```bash
git clone https://github.com/abhikr1871/Runtime_HCL.git && cd Runtime_HCL
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
pytest -q                                   # should be all green
python scripts/download_models.py           # e5 embeddings
ollama pull qwen2.5:7b-instruct             # the LLM
```
