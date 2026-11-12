"""Tests for quorate.api — strip_think, provider detection, error handling."""

import asyncio
import json

import httpx

from quorate import api
from quorate.api import (
    _anthropic,
    _antigravity_model,
    _antigravity_prompt,
    _detect_provider,
    _diagnostic_code,
    _fireworks_us_api,
    _google,
    _kimi_cli_env,
    _kimi_code_api,
    _openrouter,
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
    def test_fireworks(self):
        assert _detect_provider("accounts/fireworks/routers/kimi-k3-us") == "fireworks"

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


def test_antigravity_model_returns_none_for_unmapped_google_ids():
    unmapped = (
        "google/gemini-4-argon",
        "google/gemini-4-carbon",
        "google/gemini-3.5-flash-lite",
        "google/gemini-3.8-flash",
    )
    for model in unmapped:
        assert _antigravity_model(model, None) is None
        assert _antigravity_model(model, ReasoningEffort.HIGH) is None
    assert _antigravity_model("google/gemini-3.5-flash", None) == "Gemini 3.5 Flash (Medium)"
    assert (
        _antigravity_model("google/gemini-3.5-flash", ReasoningEffort.LOW)
        == "Gemini 3.5 Flash (Low)"
    )
    assert (
        _antigravity_model("google/gemini-3.1-pro", ReasoningEffort.MEDIUM)
        == "Gemini 3.1 Pro (High)"
    )
    assert (
        _antigravity_model("google/gemini-3.1-pro-preview", ReasoningEffort.LOW)
        == "Gemini 3.1 Pro (Low)"
    )


def test_antigravity_prompt_refuses_unmapped_model_without_subprocess(monkeypatch):
    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("unmapped model must not spawn a subprocess")

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", fail_if_called)

    async def run():
        return await _antigravity_prompt(
            "google/gemini-4-argon",
            [Message.user("question")],
            10,
            None,
        )

    result, tokens = asyncio.run(run())
    assert result == "[Error: Antigravity has no verified label for gemini-4-argon]"
    assert tokens is None
    assert _diagnostic_code(result) == "unmapped_model"


def test_google_sends_key_in_header_not_url():
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _google(
                client,
                "google-key",
                "google/gemini-3.5-flash",
                [Message.user("question")],
                128,
                10,
                None,
            )

    result, _tokens = asyncio.run(run())
    assert result == "answer"
    assert len(captured) == 1
    assert captured[0].headers["x-goog-api-key"] == "google-key"
    assert "google-key" not in str(captured[0].url)
    assert "key=" not in str(captured[0].url)


def test_google_adds_thinking_budget_to_output_cap():
    bodies = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}]
            },
        )

    async def run(effort: ReasoningEffort | None) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await _google(
                client,
                "google-key",
                "google/gemini-3.5-flash",
                [Message.user("question")],
                1024,
                10,
                effort,
            )

    asyncio.run(run(None))
    asyncio.run(run(ReasoningEffort.HIGH))
    asyncio.run(run(ReasoningEffort.LOW))

    assert [
        (
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            body["generationConfig"]["maxOutputTokens"],
        )
        for body in bodies
    ] == [(4096, 5120), (16000, 17024), (512, 1536)]


def test_google_joins_text_parts_and_skips_thoughts():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "private reasoning", "thought": True},
                                {"text": "Alpha "},
                                {"text": "beta"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _google(
                client,
                "google-key",
                "google/gemini-3.5-flash",
                [Message.user("question")],
                128,
                10,
                None,
            )

    result, _tokens = asyncio.run(run())
    assert result == "Alpha beta"


def test_google_flags_max_tokens_truncation():
    async def run(parts: list[dict], usage: dict | None):
        async def handler(_request: httpx.Request) -> httpx.Response:
            payload: dict = {
                "candidates": [{"content": {"parts": parts}, "finishReason": "MAX_TOKENS"}]
            }
            if usage is not None:
                payload["usageMetadata"] = usage
            return httpx.Response(200, json=payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _google(
                client,
                "google-key",
                "google/gemini-4-argon",
                [Message.user("question")],
                128,
                10,
                None,
            )

    text, tokens = asyncio.run(
        run([{"text": "partial answer"}], {"promptTokenCount": 12, "candidatesTokenCount": 128})
    )
    assert text == "partial answer"
    assert tokens == {"tokens_in": 12, "tokens_out": 128, "truncated": True}

    text, tokens = asyncio.run(run([{"text": "partial"}], None))
    assert text == "partial"
    assert tokens == {"truncated": True}

    text, tokens = asyncio.run(run([{"text": "private reasoning", "thought": True}], None))
    assert text == "[Error: google gemini-4-argon: max_tokens before visible text]"
    assert tokens is None
    assert _diagnostic_code(text) == "max_tokens"


def test_query_model_routes_unmapped_google_seat_to_ai_studio(monkeypatch):
    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("unmapped Google seat must not spawn a subprocess")

    monkeypatch.setattr(api.asyncio, "create_subprocess_exec", fail_if_called)

    urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await api.query_model(
                client,
                {"google": "google-key"},
                ModelEntry("Gemini-4-Argon", "google/gemini-4-argon"),
                [Message.user("question")],
            )

    result = asyncio.run(run())
    assert len(urls) == 1
    assert "gemini-4-argon:generateContent" in urls[0]
    assert result.provider == "google-ai-studio"
    assert result.model_id == "gemini-4-argon"
    assert result.response == "answer"
    assert result.diagnostics == ("antigravity-cli:unmapped_model",)


def test_query_model_records_google_truncation_diagnostic():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "partial answer"}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 128},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await api.query_model(
                client,
                {"google": "google-key"},
                ModelEntry("Gemini-4-Argon", "google/gemini-4-argon"),
                [Message.user("question")],
            )

    result = asyncio.run(run())
    assert result.provider == "google-ai-studio"
    assert result.response == "partial answer"
    assert not result.is_error
    assert result.diagnostics == (
        "antigravity-cli:unmapped_model",
        "google-ai-studio:max_tokens",
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


def test_fireworks_us_api_uses_exact_router_and_chat_completions():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.fireworks.ai/inference/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer fireworks-key"
        body = json.loads(request.content)
        assert body == {
            "model": "accounts/fireworks/routers/kimi-k3-us",
            "messages": [{"role": "user", "content": "question"}],
            "max_tokens": 128,
            "context_length_exceeded_behavior": "error",
            "service_tier": "priority",
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "private reasoning",
                            "content": "answer",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _fireworks_us_api(
                client,
                "fireworks-key",
                "accounts/fireworks/routers/kimi-k3-us",
                [Message.user("question")],
                128,
                10,
                ReasoningEffort.HIGH,
            )

    assert asyncio.run(run()) == ("answer", {"tokens_in": 12, "tokens_out": 4})


def test_fireworks_us_api_reports_malformed_json_safely():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _fireworks_us_api(
                client,
                "fireworks-key",
                "accounts/fireworks/routers/kimi-k3-us",
                [Message.user("question")],
                128,
                10,
                None,
            )

    result, tokens = asyncio.run(run())
    assert _diagnostic_code(result) == "invalid_json"
    assert tokens is None


def test_fireworks_us_api_rejects_malformed_response_shape():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": "not-a-list"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _fireworks_us_api(
                client,
                "fireworks-key",
                "accounts/fireworks/routers/kimi-k3-us",
                [Message.user("question")],
                128,
                10,
                None,
            )

    result, tokens = asyncio.run(run())
    assert _diagnostic_code(result) == "invalid_response"
    assert tokens is None


def test_fireworks_us_api_rejects_malformed_usage_tokens():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": "12", "completion_tokens": 4},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _fireworks_us_api(
                client,
                "fireworks-key",
                "accounts/fireworks/routers/kimi-k3-us",
                [Message.user("question")],
                128,
                10,
                None,
            )

    result, tokens = asyncio.run(run())
    assert _diagnostic_code(result) == "invalid_response"
    assert tokens is None


def test_fireworks_us_api_treats_reasoning_only_response_as_empty():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "private reasoning",
                            "content": "",
                        }
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _fireworks_us_api(
                client,
                "fireworks-key",
                "accounts/fireworks/routers/kimi-k3-us",
                [Message.user("question")],
                128,
                10,
                None,
            )

    result, tokens = asyncio.run(run())
    assert _diagnostic_code(result) == "empty_response"
    assert tokens is None


def test_fireworks_us_api_fails_closed_on_truncated_final_content():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "partial answer"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 128},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _fireworks_us_api(
                client,
                "fireworks-key",
                "accounts/fireworks/routers/kimi-k3-us",
                [Message.user("question")],
                128,
                10,
                None,
            )

    result, tokens = asyncio.run(run())
    assert _diagnostic_code(result) == "truncated_response"
    assert tokens == {"tokens_in": 12, "tokens_out": 128}


def test_fireworks_us_api_sanitizes_provider_error_prose():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "secret-like provider detail"}},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _fireworks_us_api(
                client,
                "fireworks-key",
                "accounts/fireworks/routers/kimi-k3-us",
                [Message.user("question")],
                128,
                10,
                None,
            )

    result, _ = asyncio.run(run())
    assert result == ("[Error: HTTP 400 from Fireworks US accounts/fireworks/routers/kimi-k3-us]")
    assert "secret-like" not in result


def test_fireworks_us_api_sanitizes_transport_error_prose():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError(
            "secret-like transport detail",
            request=request,
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _fireworks_us_api(
                client,
                "fireworks-key",
                "accounts/fireworks/routers/kimi-k3-us",
                [Message.user("question")],
                128,
                10,
                None,
            )

    result, _ = asyncio.run(run())
    assert _diagnostic_code(result) == "transport_error"
    assert "secret-like" not in result


def test_fireworks_namespace_rejects_non_us_model_without_fallback():
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("Unsupported Fireworks model must not make a request")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await api.query_model(
                client,
                {
                    "fireworks": "fireworks-key",
                    "openrouter": "fallback-key",
                },
                ModelEntry(
                    "Kimi-K3",
                    "accounts/fireworks/models/kimi-k3",
                ),
                [Message.user("question")],
            )

    result = asyncio.run(run())
    assert result.provider == "none"
    assert result.is_error
    assert result.diagnostics == ("fireworks-us:unsupported_model",)


def test_fireworks_us_route_fails_closed_without_key_even_if_openrouter_is_available():
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("Fireworks US route must not fall through to OpenRouter")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await api.query_model(
                client,
                {"openrouter": "fallback-key"},
                ModelEntry(
                    "Kimi-K3-US",
                    "accounts/fireworks/routers/kimi-k3-us",
                ),
                [Message.user("question")],
            )

    result = asyncio.run(run())
    assert result.provider == "none"
    assert result.is_error
    assert result.diagnostics == ("fireworks-us:no_credentials",)


def test_fireworks_us_route_records_successful_provider_and_usage():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await api.query_model(
                client,
                {"fireworks": "fireworks-key"},
                ModelEntry(
                    "Kimi-K3-US",
                    "accounts/fireworks/routers/kimi-k3-us",
                ),
                [Message.user("question")],
            )

    result = asyncio.run(run())
    assert result.provider == "fireworks-us"
    assert result.response == "answer"
    assert result.tokens_in == 12
    assert result.tokens_out == 4
    assert result.diagnostics == ()


def test_fireworks_us_route_fails_closed_after_http_error():
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(503)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await api.query_model(
                client,
                {
                    "fireworks": "fireworks-key",
                    "openrouter": "fallback-key",
                },
                ModelEntry(
                    "Kimi-K3-US",
                    "accounts/fireworks/routers/kimi-k3-us",
                ),
                [Message.user("question")],
            )

    result = asyncio.run(run())
    assert result.provider == "none"
    assert result.is_error
    assert result.diagnostics == ("fireworks-us:http_503",)
    assert requests == ["https://api.fireworks.ai/inference/v1/chat/completions"]


def test_fireworks_judge_route_is_rejected_as_out_of_role(monkeypatch):
    monkeypatch.setattr(
        api,
        "api_keys",
        lambda: {"fireworks": None, "openrouter": "fallback-key"},
    )

    result = asyncio.run(
        api.query_judge(
            "accounts/fireworks/routers/kimi-k3-us",
            [Message.user("question")],
        )
    )

    assert result == "[Error: Fireworks US is restricted to member seats]"


def test_fireworks_judge_route_does_not_call_any_provider(monkeypatch):
    monkeypatch.setattr(
        api,
        "api_keys",
        lambda: {"fireworks": "fireworks-key", "openrouter": "fallback-key"},
    )

    async def fail_if_provider_called(*_args, **_kwargs):
        raise AssertionError("Out-of-role Fireworks judge must not call a provider")

    monkeypatch.setattr(api, "_fireworks_us_api", fail_if_provider_called)
    monkeypatch.setattr(api, "_openrouter", fail_if_provider_called)

    result = asyncio.run(
        api.query_judge(
            "accounts/fireworks/routers/kimi-k3-us",
            [Message.user("question")],
        )
    )

    assert result == "[Error: Fireworks US is restricted to member seats]"


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
    monkeypatch.setenv("FIREWORKS_API_KEY", "secret")
    monkeypatch.setenv("QUORATE_FIREWORKS_KEY", "secret")
    environment = _subscription_cli_env()
    assert environment["PATH"] == "/bin"
    assert "OPENROUTER_API_KEY" not in environment
    assert "XAI_API_KEY" not in environment
    assert "FIREWORKS_API_KEY" not in environment
    assert "QUORATE_FIREWORKS_KEY" not in environment
    assert _kimi_cli_env() == environment


def test_anthropic_sends_adaptive_thinking_for_claude_5_family():
    bodies = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "answer"}],
                "usage": {"input_tokens": 12, "output_tokens": 4},
            },
        )

    async def run(model: str) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await _anthropic(
                client,
                "anthropic-key",
                model,
                [Message.user("question")],
                128,
                10,
                ReasoningEffort.HIGH,
            )

    asyncio.run(run("anthropic/claude-opus-5"))
    asyncio.run(run("anthropic/claude-fable-5-2"))

    assert [body["model"] for body in bodies] == ["claude-opus-5", "claude-fable-5-2"]
    for body in bodies:
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "high"}
        assert "budget_tokens" not in body["thinking"]
        assert body["max_tokens"] == 4096


def test_anthropic_adaptive_models_omit_thinking_without_effort():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "thinking" not in body
        assert "output_config" not in body
        assert body["max_tokens"] == 4096
        return httpx.Response(200, json={"content": [{"type": "text", "text": "answer"}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _anthropic(
                client,
                "anthropic-key",
                "anthropic/claude-opus-5",
                [Message.user("question")],
                128,
                10,
                None,
            )

    result, _tokens = asyncio.run(run())
    assert result == "answer"


def test_anthropic_keeps_budget_tokens_for_legacy_models():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "question"}],
            "max_tokens": 34000,
            "thinking": {"type": "enabled", "budget_tokens": 32000},
        }
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "answer"}]},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _anthropic(
                client,
                "anthropic-key",
                "anthropic/claude-haiku-4-5",
                [Message.user("question")],
                128,
                10,
                ReasoningEffort.HIGH,
            )

    result, _tokens = asyncio.run(run())
    assert result == "answer"


def test_openrouter_body_uses_dotted_anthropic_ids():
    bodies = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    async def run(model: str) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await _openrouter(
                client,
                "openrouter-key",
                model,
                [Message.user("question")],
                128,
                10,
                None,
            )

    asyncio.run(run("anthropic/claude-fable-5-2"))
    asyncio.run(run("anthropic/claude-opus-5"))

    assert [body["model"] for body in bodies] == [
        "anthropic/claude-fable-5.2",
        "anthropic/claude-opus-5",
    ]


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
