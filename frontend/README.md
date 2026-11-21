# Shopwise AI Frontend

A React + Vite frontend for the AI-powered conversational shopping assistant described in the HCLTech hackathon participant guide and architecture diagram.


## Features
- Chat workspace with demo/mock recommendations (English/Hinglish sample prompts)
- Product recommendations and reason/product facts
- Browse catalogue with category, budget, brand, stock and text filters
- Select up to 3 products for comparison
- Trace explorer with locally generated demo trace, request, pipeline steps and catalogue source fields
- JSON product ingest (local catalogue, optionally POST to FastAPI)
- API mode for `POST /chat`, `GET /health`, `POST /ingest`; backend API base can be edited in source at `apiBase` state
- Responsive desktop/tablet/mobile layout


## Requirements
Node.js 18+ recommended.


## Run locally
```bash
npm install
npm run dev
```
Open the local URL printed by Vite.\n

## Build production bundle
```bash
npm run build
npm run preview
```

## API mode
The frontend expects a backend matching the project guide's FastAPI endpoints. In `src/main.jsx`, change the `apiBase` default (currently `http://localhost:8000`) if needed. Toggle **Connect API** in the left sidebar to enable requests.


### Expected chat request
```json
{
  "session_id": "demo-session",
  "customer_id": "demo-shopper",
  "message": "Find trekking gifts under 3000"
}
```

### Expected chat response (compatible with guide)
```json
{
  "trace_id": "a41c9e07",
  "answer_type": "recommendation",
  "answer": "Here are some options.",
  "products": [
    {"product_id":"P-2041","name":"Trail Daypack 25L","price":2499,"in_stock":true,"reason":"Under budget and suited to trekking."}
  ],
  "sources": [],
  "follow_up_question": null
}
```

Note: mock mode is intended for frontend development and contains a small sample product catalogue. The participant guide recommends a minimum 250 products for your full data catalogue, 25 generated shopper profiles, and a 40-question evaluation set; those should be supplied by the backend/data kit.

## Structure
```
├── index.html
├── package.json
└── src
    ├── main.jsx
    └── styles.css
```
