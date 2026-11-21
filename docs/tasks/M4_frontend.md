# M4: Frontend and packaging (Streamlit UI, Docker, README)

**Goal:** a clean demo where judges can SEE grounding, memory and safety, and `docker compose up` just works.

**You own:** `ui/*`, `docker-compose.yml`, `api/Dockerfile`, `ui/Dockerfile`, `README.md`

## Tasks

**0:50-2:00**
- [ ] Run `streamlit run ui/app.py` against the API; fix layout of product cards; Hindi text renders correctly
- [ ] Customer picker in the sidebar (C-001 ... C-030 once M5's shoppers exist)

**2:00-3:30**
- [ ] Filter chips: removing a chip updates the next answer
- [ ] "Why these results?" panel: show the funnel as a small bar (270 → 18 → 3) and per-product sources
- [ ] Trace viewer page: readable layout (slots, retrieval scores, validator, timings)
- [ ] Admin page: CSV upload once M2 adds it

**3:30-4:30**
- [ ] `docker compose up --build` from a fresh clone; API reaches Ollama on the host
- [ ] README: run steps, curl samples, assumptions; screenshots

**4:30-6:00**
- [ ] Demo script rehearsal: normal search, vague question, follow-up, Hinglish, not-found, injection, live ingest

## Done when
- Every demo scenario is visible in the UI without opening the terminal
- A teammate can start everything from the README alone

## Be ready to explain
How the UI talks only to the API; how chips map to session state; how to run the system.
