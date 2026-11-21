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
import time
from typing import Any, Optional

import httpx

from app.config import get_settings

log = logging.getLogger("llm")
_OPENAI_COMPATIBLE = {"groq", "openai", "hosted"}
_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:(\d+)ms)?")


def _reset_seconds(headers: httpx.Headers, default: float) -> float:
    """Seconds to wait after a 429: Retry-After, else Groq's x-ratelimit-reset-tokens/-requests ('10.8s', '12m57s')."""
    if headers.get("retry-after"):
        try:
            return float(headers["retry-after"])
        except ValueError:
            pass
    for name in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        m = _DURATION.fullmatch(headers.get(name, "").strip())
        if m and any(m.groups()):
            h, mins, s, ms = m.groups()
            return int(h or 0) * 3600 + int(mins or 0) * 60 + float(s or 0) + int(ms or 0) / 1000
    return default


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


def _error_code(r: httpx.Response) -> str:
    """Provider error code (e.g. rate_limit_exceeded, json_validate_failed) for logs; never includes the key."""
    try:
        err = r.json().get("error", {})
        return str(err.get("code") or err.get("type") or "")
    except Exception:
        return ""


class LLMClient:
    def __init__(self, ignore_mock: bool = False, timeout_s: Optional[float] = None, max_wait_s: float = 2.0) -> None:
        """`ignore_mock=True` is for offline data generation, which needs the real model even when the app runs in mock mode.
        `max_wait_s` caps how long one call may wait for a rate-limit reset: keep it small for live chat
        (fall back to rules/templates fast), larger for batch data generation."""
        self.s = get_settings()
        self.mock = self.s.mock_llm and not ignore_mock
        self.timeout = timeout_s or self.s.llm_timeout_s
        self.max_wait_s = max_wait_s
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
        for attempt in range(5):
            self.calls += 1
            try:
                r = httpx.post(url, json=body, headers=headers, timeout=self.timeout)
            except httpx.HTTPError as exc:  # timeout, connection error
                log.warning("hosted LLM call failed: %s", exc.__class__.__name__)
                if attempt < 2:
                    continue
                break
            if r.status_code == 200:
                data = r.json()
                usage = data.get("usage") or {}
                self.tokens_in += usage.get("prompt_tokens", 0)
                self.tokens_out += usage.get("completion_tokens", 0)
                content = data["choices"][0]["message"].get("content")
                if content:
                    return content
                log.warning("hosted LLM returned an empty message (finish_reason=%s)", data["choices"][0].get("finish_reason"))
                break
            err = _error_code(r)
            if r.status_code == 400 and "response_format" in body and err == "json_validate_failed":
                body.pop("response_format")  # model could not produce strict JSON: retry once without JSON mode
                log.warning("JSON mode failed (json_validate_failed); retrying without it")
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = _reset_seconds(r.headers, default=2.0 * (attempt + 1)) if r.status_code == 429 else 2.0 * (attempt + 1)
                if wait > self.max_wait_s:
                    log.warning("hosted LLM %s (%s): would need to wait %.0fs, giving up", r.status_code, err, wait)
                    break
                log.warning("hosted LLM %s (%s): waiting %.1fs, then retrying", r.status_code, err, wait)
                time.sleep(wait + 0.5)
                continue
            log.warning("hosted LLM call failed: HTTP %s (%s)", r.status_code, err)
            break
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
