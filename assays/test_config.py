"""Tests for quorate.config — model resolution, display names, error detection."""

import pytest
from quorate.config import (
    ModelEntry,
    Message,
    ReasoningEffort,
    _display_name,
    is_error,
    is_thinking_model,
    model_max_tokens,
    quick_models,
    resolved_council,
    resolved_judge,
    resolved_critique,
)


class TestDisplayName:
    def test_gpt(self):
        assert _display_name("openai/gpt-5.4-pro") == "GPT-5.4-Pro"

    def test_glm(self):
        assert _display_name("z-ai/glm-5.1") == "GLM-5.1"

    def test_deepseek(self):
        assert _display_name("deepseek/deepseek-v3.2") == "DeepSeek-V3.2"

    def test_gemini_strips_preview(self):
        assert _display_name("google/gemini-3.1-pro-preview") == "Gemini-3.1-Pro"

    def test_grok(self):
        assert _display_name("x-ai/grok-4") == "Grok-4"


class TestIsThinkingModel:
    def test_gemini_pro(self):
        assert is_thinking_model("google/gemini-3.1-pro-preview") is True

    def test_gpt_54(self):
        assert is_thinking_model("openai/gpt-5.4-pro") is True

    def test_grok_420(self):
        assert is_thinking_model("x-ai/grok-4.20-0309-reasoning") is True

    def test_glm_51(self):
        assert is_thinking_model("z-ai/glm-5.1") is True

    def test_non_thinking(self):
        assert is_thinking_model("anthropic/claude-haiku-4-5") is False


class TestIsError:
    def test_error_prefix(self):
        assert is_error("[Error: something]") is True

    def test_no_response(self):
        assert is_error("[No response from model]") is True

    def test_empty(self):
        assert is_error("") is True

    def test_normal(self):
        assert is_error("Hello world") is False

    def test_bracket_not_error(self):
        assert is_error("[Some other text]") is False


class TestModelMaxTokens:
    def test_gemini_3(self):
        assert model_max_tokens("google/gemini-3.1-pro") == 65536

    def test_claude(self):
        assert model_max_tokens("anthropic/claude-opus-4-6") == 32000

    def test_gpt(self):
        assert model_max_tokens("openai/gpt-5.4-pro") == 16384

    def test_unknown(self):
        assert model_max_tokens("unknown/model") == 8192


class TestReasoningEffort:
    def test_values(self):
        assert ReasoningEffort.LOW.value == "low"
        assert ReasoningEffort.HIGH.value == "high"

    def test_anthropic_budget(self):
        assert ReasoningEffort.HIGH.anthropic_budget() == 32000

    def test_google_budget(self):
        assert ReasoningEffort.HIGH.google_budget() == 16000


class TestMessage:
    def test_system(self):
        msg = Message.system("hello")
        assert msg.role == "system"
        assert msg.content == "hello"

    def test_to_dict(self):
        msg = Message.user("test")
        assert msg.to_dict() == {"role": "user", "content": "test"}


class TestCouncilResolution:
    def test_council_returns_6(self):
        assert len(resolved_council()) == 6

    def test_quick_returns_7(self):
        assert len(quick_models()) == 7

    def test_judge_default(self):
        assert "gemini" in resolved_judge().lower()

    def test_critique_default(self):
        assert "opus" in resolved_critique().lower()
