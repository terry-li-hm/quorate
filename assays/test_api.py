"""Tests for quorate.api — strip_think, provider detection, error handling."""

import pytest
from quorate.api import _strip_think, _detect_provider


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
    def test_anthropic(self):
        assert _detect_provider("anthropic/claude-opus-4-6") == "anthropic"

    def test_claude_bare(self):
        assert _detect_provider("claude-sonnet-4-6") == "anthropic"

    def test_google(self):
        assert _detect_provider("google/gemini-3.1-pro-preview") == "google"

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
