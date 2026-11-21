# M5 runbook: build the data and run the evaluation

Everything is implemented. These are the commands, in order. Each script works **with Groq** (default) or
**offline with templates** (`--offline`), so the team is never blocked by the API.

## 0. One-time setup

```powershell
cd "$HOME\OneDrive\Desktop\Shopping_Assistant"; .venv\Scripts\activate
copy .env.example .env
```

Open `.env` and set (never commit this file, never paste the key anywhere else):

```text
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=<your new key>
LLM_REASONING_EFFORT=low
MOCK_LLM=true
```

`MOCK_LLM` only affects the running app; the data scripts always use Groq unless you pass `--offline`.

## 1. Build the data (about 10 minutes with Groq)

| Step | Command | What it does | Check |
| --- | --- | --- | --- |
| 1 | `python data_kit/fetch_dummyjson.py` | 184 DummyJSON products + reviews (cars/motorbikes excluded) | "saved 184 products" |
| 2 | `python data_kit/generate_products.py` | 81 India-focused products; Groq writes names, descriptions, tags; code sets IDs, prices, stock | "N written by the LLM" (ideally 81) |
| 3 | `python data_kit/plant_mess.py` | Mixed units, missing fields, stock 0/low, 12 Hindi/Hinglish descriptions (Groq), reviews, 3 injection reviews | summary counts printed |
| 4 | `python data_kit/generate_shoppers.py` | 30 shoppers (Groq writes names, brands, sizes) + 114 orders; edge cases C-028/029/030 | edge cases printed |
| 5 | `python data_kit/validate_data.py` | Every brief minimum checked; saves `data/processed/validation_report.txt` | ends with `OK` |
| 6 | `python data_kit/build_db.py` | Builds `data/shop.db` | "265 products ... 30 shoppers" |

If Groq is slow or fails, any step falls back to templates for the items it could not get, and says so.
To run the whole chain without Groq: add `--offline` to steps 2-4.

Then restart the API (`cd api; uvicorn app.main:app --port 8000`) and ask one question to confirm real products appear.

## 2. Hand-offs

- Tell the team: **"real data in"**. Data files are not in git: each teammate runs steps 1-6 (with `--offline` is fine for them).
- Commit only code and docs: `git add data_kit eval docs` (never `data/`, never `.env`).

## 3. Evaluation (API running on port 8000)

```powershell
python eval/run_eval.py --label mock
```

Then set `MOCK_LLM=false` in `.env`, restart the API, and:

```powershell
python eval/run_eval.py --label groq
```

`eval/report.md` is regenerated each run (keep the final one). `eval/results/<label>.json` keeps every case's details.
Comparing `mock` vs `groq` is a strong slide: "what the LLM adds, and what it costs (calls, tokens, latency)".

## 4. What the first offline run showed (template data, mock LLM)

37/54 passed · relevance@3 95% · constraint accuracy 90% · groundedness 100% · attacks succeeded 2 of 8.
Failures to hand to owners:

| Failing group | Cause | Owner |
| --- | --- | --- |
| typo / negation (0/5) | no typo tolerance; "not Nike" makes the query vague → clarification | M1 |
| Hindi product words ("कुर्ता", "प्रेशर कुकर") | keyword search can't match Hindi to English products (needs embeddings) | M2 |
| not-found (2/5) | "electric cars" matched "Electric Stove": no minimum relevance check | M2 |
| compare / "the second one" after a clarification | references only work after a recommendation turn | M3 |
| "Rahul's address" not refused; "system: admin mode" not refused | missing guard phrasings | M3 |

## 5. Fill in before the freeze

- `data_kit/DATA_NOTE.md`: counts from `validate_data.py`, the mess table from `data/processed/mess_log.json`,
  edge-case IDs from `edge_cases.json`, model and tokens from `generation_log.json`.
- `docs/ai_usage.md`: "Groq `openai/gpt-oss-20b`" for product text, translations and shopper personas.
- `data_kit/prompts/README.md`: model name per prompt.
