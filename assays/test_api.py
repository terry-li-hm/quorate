"""Tests for quorate.api — strip_think, provider detection, error handling."""

from quorate.api import _detect_provider, _diagnostic_code, _strip_think, quorum_health
from quorate.config import ModelCallResult


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


class TestSafeDiagnostics:
    def test_http_status_only(self):
        assert _diagnostic_code("[Error: HTTP 404 from a model]") == "http_404"

    def test_timeout(self):
        assert _diagnostic_code("[Error: request timed out after secret details]") == "timeout"

    def test_unknown_error_is_generic(self):
        assert _diagnostic_code("[Error: sensitive provider prose]") == "provider_error"


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
