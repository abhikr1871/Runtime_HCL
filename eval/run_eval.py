"""Run the labelled evaluation set against the running API and write eval/report.md. Owner: M5.

Run:  python eval/run_eval.py            (API must be running; API_URL env var, default localhost:8000)

Every number in the report comes from this script's output - never edit results by hand
(made-up evaluation results are a disqualification condition).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")  # 127.0.0.1, not localhost: avoids a ~2 s IPv6 delay on Windows


def run_case(case: dict) -> dict:
    session, last, latencies = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}", None, []
    for msg in case["turns"]:
        t = time.perf_counter()
        r = httpx.post(f"{API_URL}/chat", json={"session_id": session, "customer_id": case.get("customer_id", "C-001"),
                                                "message": msg}, timeout=120)
        latencies.append((time.perf_counter() - t) * 1000)
        last = r.json()
    exp, checks = case["expect"], {}
    if "answer_type" in exp:
        checks["answer_type"] = last.get("answer_type") == exp["answer_type"]
    if "max_price" in exp:
        checks["constraint"] = all((p.get("price") or 0) <= exp["max_price"] for p in last.get("products", []))
    if "language" in exp:
        checks["language"] = last.get("language") == exp["language"]
    if "not_product" in exp:
        checks["safety"] = all(p["product_id"] != exp["not_product"] for p in last.get("products", []))
    # TODO(M5): relevance@3 against labelled product IDs, groundedness via GET /products/{id}
    return {"id": case["id"], "group": case["group"], "passed": all(checks.values()), "checks": checks,
            "latency_ms": latencies, "answer_type": last.get("answer_type"), "trace_id": last.get("trace_id")}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # ₹ and Hindi on Windows consoles
    cases = [json.loads(line) for line in (ROOT / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    results = [run_case(c) for c in cases]
    lat = sorted(x for r in results for x in r["latency_ms"])
    by_group = defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r["passed"])

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    lines = ["# Evaluation report", "", f"Cases: {len(results)} · passed: {sum(r['passed'] for r in results)}", "",
             "| Group | Passed | Total |", "| --- | --- | --- |"]
    lines += [f"| {g} | {sum(v)} | {len(v)} |" for g, v in sorted(by_group.items())]
    if lat:
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        lines += ["", f"Latency: p50 {statistics.median(lat):.0f} ms · p95 {p95:.0f} ms"]
    lines += ["", "## Failures we found", ""] + [f"- `{r['id']}`: {r['checks']} (trace {r['trace_id']})"
                                                for r in results if not r["passed"]]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
