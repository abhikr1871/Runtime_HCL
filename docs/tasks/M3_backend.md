# M3: Backend (API, database, session state, privacy, traces)

**Goal:** a stable API that follows the brief's contract, remembers the conversation and never leaks another shopper's data.

**You own:** `api/app/main.py`, `config.py`, `schemas.py` (with M1), `api/app/routes/*` (except ingest), `api/app/store/db.py`, `repository.py`, `schema.sql`, `api/app/pipeline/guard.py`, `state.py`, `compare.py`, `personalise.py`, `api/app/observe/*`

## Tasks

**0:50-2:00**
- [ ] `state.py`: detect a new topic (different product type) and reset old filters
- [ ] `state.py`: "the blue one", "that brand", "same but in black" against the last results
- [ ] Load shoppers and orders tables (with M5's `build_db.py`)

**2:00-3:30**
- [ ] `guard.py`: more privacy phrasings incl. Hindi script; output guard scanning replies for other customers' IDs/names
- [ ] `personalise.py`: brand/size tie-break from the shopper's OWN orders, stated as an assumption
- [ ] Out-of-stock favourite → substitute message (`filters.out_of_stock_matches`)
- [ ] `compare.py`: category-specific attributes in the table
- [ ] 3 sample audit records exported to `docs/audit_samples/`

**3:30-4:30**
- [ ] Help M4 with Docker Compose; check `docker compose up` from a fresh clone

## Done when
- `tests/test_privacy.py` green, plus new tests for each privacy phrasing you add
- `tests/test_api.py::test_follow_up_cheaper` and `test_compare_first_two` pass with real data
- Every reply has a trace that replays via `GET /trace/{id}`

## Be ready to explain
Privacy by construction (queries only take the session owner's ID); how the trace is built; why identity never comes from the message.
