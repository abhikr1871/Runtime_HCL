"""One client for every LLM call. Owner: M1.

Providers: mock (no calls), ollama (local), hosted (organiser-approved only; disclose in README).
Every method returns None on any failure, so callers ALWAYS keep a rules/template fallback.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

from app.config import get_settings

log = logging.getLogger("llm")


class LLMClient:
    def __init__(self) -> None:
        self.s = get_settings()
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    @property
    def name(self) -> str:
        return "mock" if self.s.mock_llm else f"{self.s.llm_provider}:{self.s.llm_model}"

    def _ollama_chat(self, system: str, user: str, json_mode: bool) -> Optional[str]:
        body: dict[str, Any] = {
            "model": self.s.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if json_mode:
            body["format"] = "json"
        try:
            self.calls += 1
            r = httpx.post(f"{self.s.ollama_url}/api/chat", json=body, timeout=self.s.llm_timeout_s)
            r.raise_for_status()
            data = r.json()
            self.tokens_in += data.get("prompt_eval_count", 0)
            self.tokens_out += data.get("eval_count", 0)
            return data["message"]["content"]
        except Exception as exc:  # timeout, connection refused, bad response
            log.warning("ollama call failed: %s", exc)
            return None

    def complete_text(self, system: str, user: str) -> Optional[str]:
        if self.s.mock_llm:
            return None
        if self.s.llm_provider == "ollama":
            return self._ollama_chat(system, user, json_mode=False)
        # TODO(M1): hosted provider behind the same interface, only if organisers approve it
        return None

    def complete_json(self, system: str, user: str) -> Optional[dict[str, Any]]:
        if self.s.mock_llm:
            return None
        raw = self._ollama_chat(system, user, json_mode=True) if self.s.llm_provider == "ollama" else None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.warning("LLM returned invalid JSON")
            return None

    def ping(self) -> str:
        if self.s.mock_llm:
            return "mock"
        try:
            httpx.get(f"{self.s.ollama_url}/api/tags", timeout=3).raise_for_status()
            return "ok"
        except Exception:
            return "unreachable"
