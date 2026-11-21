# Frontend review (React UI, merged into main)

The React UI in `frontend/` is merged. These changes are needed before the demo. Owner: Ayush (M4).

| # | Issue | Fix |
| --- | --- | --- |
| 0 | **Crash (fixed in main):** `useEffect(()=>bottomRef.current?.scrollIntoView(...))` returned a Promise in current Chrome, so React threw "destroy is not a function" and the page went blank after the first message. Fixed by wrapping the body in `{ }`. | Keep effects as `useEffect(() => { ... }, deps)` |
| 0b | In API mode every product shows "1 left", because `/chat` returns `in_stock` (true/false), not a stock count. Products with no brand show "Catalog". | Show "In stock" / "Out of stock" from `in_stock`; show "Brand not listed" for missing brands (the catalogue has 5 on purpose) |
| 1 | It uses a hardcoded demo catalogue (16 made-up products, `P-2041` ...) and starts in **Demo mode**. Judges must see our real 265-product catalogue. | Start in API mode by default; remove `PRODUCTS` / `mockAnswer` or keep them only as an offline fallback clearly labelled "demo". Load products from `GET /products/{id}` results returned by `/chat`. |
| 2 | API base is `http://localhost:8000`. On Windows `localhost` adds ~2 s per request. | Use `http://127.0.0.1:8000`. |
| 3 | `customer_id` is hardcoded to `demo-shopper` and `session_id` to `demo-session` (every browser tab shares one session). | Add a customer picker (C-001 ... C-030) and a fresh `session_id` per conversation (e.g. `crypto.randomUUID()`). |
| 4 | Ingest sends `price`; the API expects `price_inr`. | Send the API's field names (`product_id`, `name`, `category`, `brand`, `price_inr`, `stock`, `rating`, `tags`, `description`). |
| 5 | The trace page shows a locally made-up trace. | Fetch the real one: `GET /trace/{trace_id}` (slots, retrieval scores, validator, timings). |
| 6 | Package versions are `"latest"`. | Pin exact versions (`npm install` then copy versions from `package-lock.json` into `package.json`). |
| 7 | Not in Docker / README. | If the team keeps React, add a `frontend` service to `docker-compose.yml` and run steps to the README. Team decision: React **or** Streamlit (`ui/`) for the demo - not both half-done. |

CORS is already handled on the API side (`CORS_ORIGINS`, default allows `http://localhost:5173` and `http://127.0.0.1:5173`).

## How to run it against the real API

```powershell
# terminal 1 (repo root)
.venv\Scripts\activate; python data_kit/build_db.py; cd api; uvicorn app.main:app --port 8000
# terminal 2
cd frontend; npm install; npm run dev     # open http://localhost:5173 and switch to API mode
```
