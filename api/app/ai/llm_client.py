"""One client for every LLM call. Owner: M1.

Providers:
- mock    : MOCK_LLM=true, no calls at all (rules + templates)
- ollama  : local model via Ollama
- groq    : Groq cloud (OpenAI-compatible API), key in GROQ_API_KEY
- openai  : any other OpenAI-compatible API (HOSTED_API_URL + HOSTED_API_KEY)

Every method returns None on any failure, so callers ALWAYS keep a rules/template fallback.
The API key is never logged.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import httpx

from app.config import get_settings

log = logging.getLogger("llm")
_OPENAI_COMPATIBLE = {"groq", "openai", "hosted"}


def parse_json_loose(raw: str) -> Optional[Any]:
    """Parse JSON, tolerating ```json fences or text around a single JSON object/array."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
    return None


class LLMClient:
    def __init__(self, ignore_mock: bool = False, timeout_s: Optional[float] = None) -> None:
        """`ignore_mock=True` is for offline data generation, which needs the real model even when the app runs in mock mode."""
        self.s = get_settings()
        self.mock = self.s.mock_llm and not ignore_mock
        self.timeout = timeout_s or self.s.llm_timeout_s
        self.calls = 0
        self.failures = 0
        self.tokens_in = 0
        self.tokens_out = 0

    @property
    def name(self) -> str:
        return "mock" if self.mock else f"{self.s.llm_provider}:{self.s.llm_model}"

    # ---------- providers ----------
    def _ollama_chat(self, system: str, user: str, json_mode: bool, temperature: float) -> Optional[str]:
        body: dict[str, Any] = {
            "model": self.s.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            body["format"] = "json"
        self.calls += 1
        try:
            r = httpx.post(f"{self.s.ollama_url}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            self.tokens_in += data.get("prompt_eval_count", 0)
            self.tokens_out += data.get("eval_count", 0)
            return data["message"]["content"]
        except Exception as exc:  # timeout, connection refused, bad response
            self.failures += 1
            log.warning("ollama call failed: %s", exc.__class__.__name__)
            return None

    def _openai_chat(self, system: str, user: str, json_mode: bool, temperature: float,
                     max_tokens: Optional[int]) -> Optional[str]:
        if not self.s.hosted_api_key or not self.s.hosted_api_url:
            log.warning("hosted LLM not configured (missing API key or URL)")
            return None
        body: dict[str, Any] = {
            "model": self.s.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if self.s.llm_reasoning_effort:
            body["reasoning_effort"] = self.s.llm_reasoning_effort
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.s.hosted_api_key}"}
        url = f"{self.s.hosted_api_url.rstrip('/')}/chat/completions"
        for attempt in range(2):
            self.calls += 1
            try:
                r = httpx.post(url, json=body, headers=headers, timeout=self.timeout)
                if r.status_code == 400 and "response_format" in body:
                    body.pop("response_format")  # model without JSON mode: retry, parse loosely
                    continue
                if r.status_code == 429 and attempt == 0:
                    continue  # rate limited: one immediate retry
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage") or {}
                self.tokens_in += usage.get("prompt_tokens", 0)
                self.tokens_out += usage.get("completion_tokens", 0)
                return data["choices"][0]["message"].get("content")
            except Exception as exc:
                self.failures += 1
                log.warning("hosted LLM call failed: %s", exc.__class__.__name__)
                return None
        self.failures += 1
        return None

    def _chat(self, system: str, user: str, json_mode: bool, temperature: float, max_tokens: Optional[int]) -> Optional[str]:
        if self.mock:
            return None
        if self.s.llm_provider == "ollama":
            return self._ollama_chat(system, user, json_mode, temperature)
        if self.s.llm_provider in _OPENAI_COMPATIBLE:
            return self._openai_chat(system, user, json_mode, temperature, max_tokens)
        log.warning("unknown LLM_PROVIDER %r", self.s.llm_provider)
        return None

    # ---------- public API ----------
    def complete_text(self, system: str, user: str, temperature: float = 0.1,
                      max_tokens: Optional[int] = None) -> Optional[str]:
        return self._chat(system, user, False, temperature, max_tokens)

    def complete_json(self, system: str, user: str, temperature: float = 0.1,
                      max_tokens: Optional[int] = None) -> Optional[Any]:
        raw = self._chat(system, user, True, temperature, max_tokens)
        if raw is None:
            return None
        parsed = parse_json_loose(raw)
        if parsed is None:
            self.failures += 1
            log.warning("LLM returned invalid JSON")
        return parsed

    def ping(self) -> str:
        if self.mock:
            return "mock"
        try:
            if self.s.llm_provider == "ollama":
                httpx.get(f"{self.s.ollama_url}/api/tags", timeout=3).raise_for_status()
            else:
                httpx.get(f"{self.s.hosted_api_url.rstrip('/')}/models", timeout=5,
                          headers={"Authorization": f"Bearer {self.s.hosted_api_key}"}).raise_for_status()
            return "ok"
        except Exception:
            return "unreachable"
