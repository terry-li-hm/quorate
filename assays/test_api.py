"""Tests for quorate.api — strip_think, provider detection, error handling."""

import asyncio
import json

import httpx

from quorate import api
from quorate.api import (
    _antigravity_model,
    _detect_provider,
    _diagnostic_code,
    _kimi_cli_env,
    _kimi_code_api,
    _parse_kimi_stream,
    _strip_think,
    _subscription_cli_env,
    quorum_health,
)
from quorate.config import Message, ModelCallResult, ModelEntry, ReasoningEffort


class TestStripThink:
    def test_basic(self):
        assert _strip_think("Hello <think>internal</think> World") == "Hello  World"

    def test_multiline(self):
        assert _strip_think("Before\n<think>\nline1\nline2\n</think>\nAfter") == "Before\n\nAfter"

    def test_no_think(self):
        assert _strip_think("Normal content") == "Normal content"

    def test_multiple(self):
        assert _strip_think("<think>first</think>middle<think>second</think>end") == "middleend"


class TestDetectProvider:
    def test_kimi_code(self):
        assert _detect_provider("kimi-code/k3") == "kimi-code"

    def test_anthropic(self):
        assert _detect_provider("anthropic/claude-opus-4-6") == "anthropic"

    def test_claude_bare(self):
        assert _detect_provider("claude-sonnet-4-6") == "anthropic"

    def test_google(self):
        assert _detect_provider("google/gemini-3.5-flash") == "google"

    def test_xai(self):
        assert _detect_provider("x-ai/grok-4") == "xai"

    def test_openai(self):
        assert _detect_provider("openai/gpt-5.4-pro") == "openai"

    def test_deepseek_openrouter(self):
        assert _detect_provider("deepseek/deepseek-v3.2") == "openrouter"

    def test_glm_zhipu(self):
        assert _detect_provider("z-ai/glm-5.1") == "zhipu"

    def test_zhipu_bare(self):
        assert _detect_provider("zhipu/glm-4-flash") == "zhipu"

    def test_qwen_openrouter(self):
        assert _detect_provider("qwen/qwen3.6-plus") == "openrouter"


class TestAntigravityModel:
    def test_flash_defaults_to_medium(self):
        assert _antigravity_model("google/gemini-3.5-flash", None) == "Gemini 3.5 Flash (Medium)"

    def test_flash_high(self):
        assert (
            _antigravity_model("google/gemini-3.5-flash", ReasoningEffort.HIGH)
            == "Gemini 3.5 Flash (High)"
        )

    def test_pro_medium_uses_high(self):
        assert (
            _antigravity_model("google/gemini-3.1-pro", ReasoningEffort.MEDIUM)
            == "Gemini 3.1 Pro (High)"
        )

    def test_pro_low(self):
        assert (
            _antigravity_model("google/gemini-3.1-pro-preview", ReasoningEffort.LOW)
            == "Gemini 3.1 Pro (Low)"
        )


class TestSafeDiagnostics:
    def test_http_status_only(self):
        assert _diagnostic_code("[Error: HTTP 404 from a model]") == "http_404"

    def test_timeout(self):
        assert _diagnostic_code("[Error: request timed out after secret details]") == "timeout"

    def test_unknown_error_is_generic(self):
        assert _diagnostic_code("[Error: sensitive provider prose]") == "provider_error"


def test_parse_kimi_stream_excludes_metadata():
    payload = "\n".join(
        (
            '{"role":"assistant","content":"answer"}',
            '{"role":"meta","content":"resume hint"}',
        )
    )
    assert _parse_kimi_stream(payload) == "answer"


def test_kimi_code_api_uses_documented_endpoint_and_real_identity():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.kimi.com/coding/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer membership-key"
        assert request.headers["user-agent"] == "quorate"
        body = json.loads(request.content)
        assert body["model"] == "k3"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _kimi_code_api(
                client,
                "membership-key",
                "kimi-code/k3",
                [Message.user("question")],
                128,
                10,
            )

    assert asyncio.run(run()) == ("answer", {"tokens_in": 12, "tokens_out": 4})


def test_headless_kimi_fails_closed_without_membership_api_key(monkeypatch):
    monkeypatch.setattr(api.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(api.sys.stdout, "isatty", lambda: False)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: None)) as client:
            return await api.query_model(
                client,
                {},
                ModelEntry("Kimi-K3", "kimi-code/k3"),
                [Message.user("question")],
            )

    result = asyncio.run(run())
    assert result.provider == "none"
    assert result.is_error
    assert result.diagnostics == (
        "kimi-code-api:no_credentials",
        "kimi-code:interactive_only",
    )


def test_subscription_cli_environment_excludes_provider_secrets(monkeypatch):
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("XAI_API_KEY", "secret")
    environment = _subscription_cli_env()
    assert environment["PATH"] == "/bin"
    assert "OPENROUTER_API_KEY" not in environment
    assert "XAI_API_KEY" not in environment
    assert _kimi_cli_env() == environment


class TestQuorumHealth:
    @staticmethod
    def _result(ok: bool) -> ModelCallResult:
        return ModelCallResult(
            name="Model",
            model_id="model",
            response="ok" if ok else "[Error: failed]",
        )

    def test_strict_majority_for_seven(self):
        results = [self._result(True) for _ in range(4)] + [self._result(False) for _ in range(3)]
        assert quorum_health(results) == (4, 4, True)

    def test_three_of_six_is_not_a_quorum(self):
        results = [self._result(True) for _ in range(3)] + [self._result(False) for _ in range(3)]
        assert quorum_health(results) == (3, 4, False)
