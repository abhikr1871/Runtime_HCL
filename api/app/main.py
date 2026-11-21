"""FastAPI entry point. Run from the api/ folder:  uvicorn app.main:app --reload"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import chat, health, ingest, products, sessions, trace
from app.store import db, indexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    db.seed_sample_if_empty()
    log.info("index ready: %s products", indexer.rebuild())
    # load the embedding model now, so the first shopper question is not the one that waits ~25 s for it
    from app.ai import embeddings
    if embeddings.embed_query("warm up") is not None:
        log.info("embedding model warmed up")
    yield


app = FastAPI(title="AI Shopping Assistant", version="0.1.0", lifespan=lifespan)

# Browser UIs on another port (React/Vite dev server on 5173) need CORS to call the API.
# Only these local origins are allowed; set CORS_ORIGINS (comma-separated) to change them.
_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-Customer-Id"])


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    trace_id = uuid.uuid4().hex[:8]
    log.exception("unhandled error trace_id=%s path=%s", trace_id, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong, please retry.", "trace_id": trace_id})


for r in (chat.router, products.router, ingest.router, trace.router, health.router, sessions.router):
    app.include_router(r)
