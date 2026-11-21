"""LLM client: Groq/OpenAI-compatible request format and fallbacks, tested without any network call."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

import httpx  # noqa: E402

from app.ai import llm_client  # noqa: E402
from app.ai.llm_client import LLMClient, parse_json_loose  # noqa: E402
from app.config import Settings  # noqa: E402


def _settings(**over):
    base = dict(llm_provider="groq", llm_model="openai/gpt-oss-20b", ollama_url="", hosted_api_url="https://api.groq.com/openai/v1",
                hosted_api_key="test-key", llm_reasoning_effort="low", llm_timeout_s=5, mock_llm=False, embed_model="",
                enable_embeddings=False, enable_reranker=False, db_path=Path("x.db"), models_dir=Path("m"),
                sample_products=Path("s.json"))
    base.update(over)
    return Settings(**base)


def test_parse_json_loose():
    assert parse_json_loose('{"a": 1}') == {"a": 1}
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('Sure! {"a": [1, 2]} hope this helps') == {"a": [1, 2]}
    assert parse_json_loose("not json") is None


def test_groq_request_and_usage(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json, headers=headers)
        return httpx.Response(200, request=httpx.Request("POST", url),
                              json={"choices": [{"message": {"content": '{"ok": true}'}}],
                                    "usage": {"prompt_tokens": 12, "completion_tokens": 3}})

    monkeypatch.setattr(llm_client, "get_settings", lambda: _settings())
    monkeypatch.setattr(llm_client.httpx, "post", fake_post)
    c = LLMClient()
    assert c.complete_json("sys", "user") == {"ok": True}
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["body"]["model"] == "openai/gpt-oss-20b"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["reasoning_effort"] == "low"
    assert (c.calls, c.tokens_in, c.tokens_out) == (1, 12, 3)


def test_failure_returns_none(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(llm_client, "get_settings", lambda: _settings())
    monkeypatch.setattr(llm_client.httpx, "post", boom)
    c = LLMClient()
    assert c.complete_text("sys", "user") is None
    assert c.failures == 1


def test_reset_seconds_parsing():
    h = httpx.Headers
    assert llm_client._reset_seconds(h({"retry-after": "3"}), 1) == 3
    assert abs(llm_client._reset_seconds(h({"x-ratelimit-reset-tokens": "10.799s"}), 1) - 10.799) < 1e-6
    assert abs(llm_client._reset_seconds(h({"x-ratelimit-reset-tokens": "12m57.5s"}), 1) - 777.5) < 1e-6
    assert llm_client._reset_seconds(h({"x-ratelimit-reset-tokens": "250ms"}), 1) == 0.25
    assert llm_client._reset_seconds(h({}), 7) == 7


def test_rate_limit_waits_then_succeeds(monkeypatch):
    responses = [
        httpx.Response(429, headers={"x-ratelimit-reset-tokens": "1.5s"}, json={"error": {"code": "rate_limit_exceeded"}}),
        httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}], "usage": {}}),
    ]
    slept = []
    monkeypatch.setattr(llm_client, "get_settings", lambda: _settings())
    monkeypatch.setattr(llm_client.httpx, "post", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(llm_client.time, "sleep", slept.append)
    c = LLMClient(max_wait_s=10)
    assert c.complete_text("s", "u") == "hello"
    assert slept and slept[0] >= 1.5


def test_rate_limit_too_long_gives_up_fast(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: _settings())
    monkeypatch.setattr(llm_client.httpx, "post", lambda *a, **k: httpx.Response(
        429, headers={"x-ratelimit-reset-tokens": "40s"}, json={"error": {"code": "rate_limit_exceeded"}}))
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("should not wait")))
    assert LLMClient(max_wait_s=2).complete_text("s", "u") is None   # live chat: fall back to templates, don't hang


def test_json_validate_failed_retries_without_json_mode(monkeypatch):
    bodies = []

    def fake_post(url, json=None, headers=None, timeout=None):
        bodies.append(dict(json))
        if "response_format" in json:
            return httpx.Response(400, json={"error": {"code": "json_validate_failed"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": 'Here: {"a": 1}'}}], "usage": {}})

    monkeypatch.setattr(llm_client, "get_settings", lambda: _settings())
    monkeypatch.setattr(llm_client.httpx, "post", fake_post)
    assert LLMClient().complete_json("s", "u") == {"a": 1}
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]


def test_mock_makes_no_calls(monkeypatch):
    monkeypatch.setattr(llm_client, "get_settings", lambda: _settings(mock_llm=True))
    monkeypatch.setattr(llm_client.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert LLMClient().complete_text("s", "u") is None
    assert LLMClient(ignore_mock=True).mock is False
