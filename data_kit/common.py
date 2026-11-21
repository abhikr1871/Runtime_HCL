"""Shared helpers for the data kit. Owner: M5."""
from __future__ import annotations

import json
import random
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROMPTS = Path(__file__).resolve().parent / "prompts"
REFERENCE_DATE = date(2026, 11, 21)   # fixed "today" so the data is reproducible
SEED = 2026

sys.path.insert(0, str(ROOT / "api"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # ₹ and Hindi on Windows consoles

from app.ai.llm_client import LLMClient  # noqa: E402
from app.config import get_settings  # noqa: E402


def rng(salt: str) -> random.Random:
    """Deterministic random generator per script/purpose."""
    return random.Random(f"{SEED}-{salt}")


def get_llm(offline: bool) -> Optional[LLMClient]:
    """Real LLM for data generation (ignores MOCK_LLM). None = offline/template mode."""
    if offline:
        print("offline mode: templates only, no LLM calls")
        return None
    s = get_settings()
    client = LLMClient(ignore_mock=True, timeout_s=90)
    if client.ping() != "ok":
        print(f"WARNING: LLM ({s.llm_provider}:{s.llm_model}) not reachable; falling back to templates")
        return None
    print(f"using LLM {client.name}")
    return client


def load_prompt(name: str) -> str:
    """Prompts are stored verbatim in data_kit/prompts/ and loaded from there, so what we submit is what ran."""
    return (PROMPTS / name).read_text(encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def batched(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def price_point(value: float) -> int:
    """Round to an Indian retail price point ending in 9 (e.g. 2493 -> 2499)."""
    v = int(round(value))
    return max(99, (v // 100) * 100 + 99 if v >= 1000 else (v // 10) * 10 + 9)


def log_generation(script: str, llm: Optional[LLMClient], prompt_files: Iterable[str], **counts: Any) -> None:
    """Appends a record to data/processed/generation_log.json (used for DATA_NOTE.md and ai_usage.md)."""
    path = PROCESSED / "generation_log.json"
    log = read_json(path, [])
    log.append({
        "script": script,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": llm.name if llm else "templates (offline)",
        "prompt_files": list(prompt_files),
        "llm_calls": llm.calls if llm else 0,
        "llm_failures": llm.failures if llm else 0,
        "tokens_in": llm.tokens_in if llm else 0,
        "tokens_out": llm.tokens_out if llm else 0,
        **counts,
    })
    write_json(path, log)


def load_catalogue() -> list[dict]:
    """All processed products (DummyJSON + generated after mess planting)."""
    return [p for f in sorted(PROCESSED.glob("*_products.json")) for p in read_json(f, [])]
