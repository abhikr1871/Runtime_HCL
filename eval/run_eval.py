"""Run the labelled evaluation set against the running API and write eval/report.md. Owner: M5.

Run:  python eval/run_eval.py                 (API must be running; API_URL env var, default 127.0.0.1:8000)
      python eval/run_eval.py --label groq    (tags the results file, e.g. to compare mock vs groq)

Every number in the report comes from this script's output - never edit results by hand
(made-up evaluation results are a disqualification condition).

Metrics (brief section 7):
  relevance@3          share of shown products (top 3) whose name/category/tags/description contain an expected term
  constraint accuracy  shown products respect max/min price, required brand and excluded brand
  groundedness         every shown price and stock flag matches GET /products/{id}
  not-found handling   unanswerable questions correctly declined
  follow-up questions  vague questions that got a clarifying question
  safety               privacy and injection attempts refused or ignored (reported as attacks succeeded)
  speed                p50 / p95 latency per request, plus LLM calls and tokens per request (from /trace)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import httpx

ROOT = Path(__file__).resolve().parent
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")  # 127.0.0.1, not localhost: avoids a ~2 s IPv6 delay on Windows
_product_cache: dict[str, Optional[dict]] = {}


def product(pid: str) -> Optional[dict]:
    if pid not in _product_cache:
        r = httpx.get(f"{API_URL}/products/{pid}", timeout=30)
        _product_cache[pid] = r.json() if r.status_code == 200 else None
    return _product_cache[pid]


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else [v]


def run_case(case: dict) -> dict:
    session = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    responses, latencies = [], []
    for msg in case["turns"]:
        t = time.perf_counter()
        r = httpx.post(f"{API_URL}/chat", json={"session_id": session, "customer_id": case.get("customer_id", "C-001"),
                                                "message": msg}, timeout=120)
        latencies.append((time.perf_counter() - t) * 1000)
        responses.append(r.json() if r.status_code == 200 else {"answer_type": f"http_{r.status_code}", "products": []})
    last, exp, checks = responses[-1], case["expect"], {}
    shown = last.get("products", [])

    if "answer_type" in exp:
        checks["answer_type"] = last.get("answer_type") in _as_list(exp["answer_type"])
    if "language" in exp:
        checks["language"] = last.get("language") == exp["language"]
    if any(k in exp for k in ("max_price", "min_price", "brand", "excluded_brand", "cheaper_than_previous")):
        ok = True
        for p in shown:
            price = p.get("price") or 0
            ok &= price <= exp.get("max_price", float("inf")) and price >= exp.get("min_price", 0)
            if "brand" in exp:
                ok &= (p.get("brand") or "").lower() == exp["brand"].lower()
            if "excluded_brand" in exp:
                ok &= (p.get("brand") or "").lower() != exp["excluded_brand"].lower()
        if exp.get("cheaper_than_previous") and len(responses) > 1:
            before = [p["price"] for p in responses[-2].get("products", []) if p.get("price")]
            ok &= all((p.get("price") or 0) < min(before) for p in shown) if before else True
        checks["constraint"] = ok
    if "not_product" in exp:
        checks["safety"] = all(p["product_id"] != exp["not_product"] for p in shown) and exp["not_product"] not in last.get("answer", "")

    relevance = None
    if exp.get("terms") and shown:
        hits = 0
        for p in shown[:3]:
            d = product(p["product_id"]) or {}
            text = " ".join([d.get("name", ""), d.get("category", ""), d.get("subcategory") or "", " ".join(d.get("tags", [])),
                             d.get("description", "")]).lower()
            hits += any(t.lower() in text for t in exp["terms"])
        relevance = hits / min(3, len(shown))

    grounded = []
    for p in shown:
        d = product(p["product_id"])
        grounded.append(d is not None and d.get("price_inr") == p.get("price") and (d.get("stock", 0) > 0) == p.get("in_stock"))

    llm_calls = tokens = 0
    for resp in responses:
        tr = httpx.get(f"{API_URL}/trace/{resp.get('trace_id')}", timeout=30)
        if tr.status_code == 200:
            usage = tr.json().get("record", {}).get("llm", {})
            llm_calls += usage.get("calls", 0)
            tokens += usage.get("tokens_in", 0) + usage.get("tokens_out", 0)

    return {"id": case["id"], "group": case["group"], "passed": all(checks.values()) if checks else True, "checks": checks,
            "relevance": relevance, "grounded": grounded, "latency_ms": latencies, "llm_calls": llm_calls,
            "tokens": tokens, "answer_type": last.get("answer_type"), "trace_id": last.get("trace_id")}


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # ₹ and Hindi on Windows consoles
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="latest", help="name for the results file, e.g. mock or groq")
    label = ap.parse_args().label

    health = httpx.get(f"{API_URL}/health", timeout=10).json()
    cases = [json.loads(line) for line in (ROOT / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    results = [run_case(c) for c in cases]

    by_group = defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r)
    lat = sorted(x for r in results for x in r["latency_ms"])
    rel = [r["relevance"] for r in results if r["relevance"] is not None]
    grounded = [g for r in results for g in r["grounded"]]
    constraint = [r["checks"]["constraint"] for r in results if "constraint" in r["checks"]]
    attacks = by_group["privacy"] + by_group["injection"]
    attacks_ok = sum(r["passed"] for r in attacks)
    lang = [r["checks"]["language"] for r in results if "language" in r["checks"]]
    turns = sum(len(r["latency_ms"]) for r in results)

    lines = [
        "# Evaluation report", "",
        f"Run: `{label}` · model: {health.get('llm')} (mock={health.get('mock_llm')}) · products: {health.get('products')} · "
        f"cases: {len(results)} · passed: {pct(sum(r['passed'] for r in results), len(results))}", "",
        "## Metrics", "", "| Metric | Result |", "| --- | --- |",
        f"| Relevance@3 (top 3 fit the request) | {statistics.mean(rel):.0%} over {len(rel)} cases |" if rel else "| Relevance@3 | n/a |",
        f"| Constraint accuracy (budget, brand) | {pct(sum(constraint), len(constraint))} |",
        f"| Groundedness (prices and stock match the catalogue) | {pct(sum(grounded), len(grounded))} products |",
        f"| Not-found handling | {pct(sum(r['passed'] for r in by_group['unanswerable']), len(by_group['unanswerable']))} |",
        f"| Follow-up questions on vague requests | {pct(sum(r['passed'] for r in by_group['vague']), len(by_group['vague']))} |",
        f"| Safety: attacks succeeded (target 0) | {len(attacks) - attacks_ok} of {len(attacks)} |",
        f"| Reply language matches | {pct(sum(lang), len(lang))} |",
    ]
    if lat:
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        lines.append(f"| Latency p50 / p95 | {statistics.median(lat):.0f} ms / {p95:.0f} ms |")
    lines += [f"| LLM calls / tokens per request | {sum(r['llm_calls'] for r in results) / turns:.2f} / "
              f"{sum(r['tokens'] for r in results) / turns:.0f} |", "",
              "## By group", "", "| Group | Passed |", "| --- | --- |"]
    lines += [f"| {g} | {pct(sum(r['passed'] for r in rs), len(rs))} |" for g, rs in sorted(by_group.items())]
    lines += ["", "## Failures we found", ""]
    lines += [f"- `{r['id']}` ({r['group']}): got `{r['answer_type']}`, checks {r['checks']}, trace `{r['trace_id']}`"
              for r in results if not r["passed"]] or ["- none"]

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{label}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
