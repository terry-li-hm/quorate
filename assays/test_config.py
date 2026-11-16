"""Tests for quorate.config — model resolution, display names, error detection."""

from quorate.config import (
    CLASSIFIER_MODEL,
    Message,
    ModelEntry,
    ReasoningEffort,
    _display_name,
    _normalize_model,
    anthropic_uses_adaptive_thinking,
    api_keys,
    benchmark_models,
    brainstorm_models,
    is_error,
    is_thinking_model,
    model_max_tokens,
    openrouter_model_id,
    quick_models,
    resolved_council,
    resolved_critique,
    resolved_critique_fallback,
    resolved_judge,
    resolved_judge_fallback,
)


def test_api_keys_include_kimi_code_membership_key(monkeypatch):
    monkeypatch.setenv("KIMI_CODE_API_KEY", "membership-key")
    assert api_keys()["kimi_code"] == "membership-key"


def test_fireworks_api_key_prefers_quorate_namespace(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "general-key")
    monkeypatch.setenv("QUORATE_FIREWORKS_KEY", "quorate-key")
    assert api_keys()["fireworks"] == "quorate-key"


class TestDisplayName:
    def test_gpt(self):
        assert _display_name("openai/gpt-5.4-pro") == "GPT-5.4-Pro"

    def test_glm(self):
        assert _display_name("z-ai/glm-5.1") == "GLM-5.1"

    def test_deepseek(self):
        assert _display_name("deepseek/deepseek-v3.2") == "DeepSeek-V3.2"

    def test_minimax(self):
        assert _display_name("minimax/minimax-m3") == "MiniMax-M3"

    def test_gemini_stable(self):
        assert _display_name("google/gemini-3.5-flash") == "Gemini-3.5-Flash"

    def test_grok(self):
        assert _display_name("x-ai/grok-4") == "Grok-4"

    def test_fireworks_kimi_k3_us(self):
        assert _display_name("accounts/fireworks/routers/kimi-k3-us") == "Kimi-K3-US"

    def test_claude_fable_5_1(self):
        assert _display_name("anthropic/claude-fable-5-1") == "Claude-Fable-5-1"

    def test_claude_fable_5_2(self):
        assert _display_name("anthropic/claude-fable-5-2") == "Claude-Fable-5-2"


class TestModelAliases:
    def test_sonnet_alias(self):
        assert _normalize_model("sonnet") == "anthropic/claude-sonnet-5"

    def test_opus_alias(self):
        assert _normalize_model("opus") == "anthropic/claude-opus-5-5"

    def test_fable_alias(self):
        assert _normalize_model("fable") == "anthropic/claude-fable-5-2"

    def test_alias_matching_ignores_case_and_padding(self):
        assert _normalize_model("  FABLE ") == "anthropic/claude-fable-5-2"


class TestIsThinkingModel:
    def test_gemini_flash(self):
        assert is_thinking_model("google/gemini-3.5-flash") is True

    def test_gpt_54(self):
        assert is_thinking_model("openai/gpt-5.4-pro") is True

    def test_grok_420(self):
        assert is_thinking_model("x-ai/grok-4.20-0309-reasoning") is True

    def test_glm_51(self):
        assert is_thinking_model("z-ai/glm-5.1") is True

    def test_deepseek_v4_pro(self):
        assert is_thinking_model("deepseek/deepseek-v4-pro") is True

    def test_minimax_m3(self):
        assert is_thinking_model("minimax/minimax-m3") is True

    def test_fireworks_kimi_k3_us(self):
        assert is_thinking_model("accounts/fireworks/routers/kimi-k3-us") is True

    def test_claude_5_family(self):
        for model in ("claude-fable-5-1", "claude-fable-5-2", "claude-opus-5", "claude-sonnet-5"):
            assert is_thinking_model(f"anthropic/{model}") is True

    def test_non_thinking(self):
        assert is_thinking_model("anthropic/claude-haiku-4-5") is False


class TestAnthropicAdaptiveThinking:
    def test_claude_5_family_is_adaptive(self):
        for model in (
            "anthropic/claude-fable-5",
            "anthropic/claude-fable-5-1",
            "anthropic/claude-fable-5-2",
            "anthropic/claude-mythos-5",
            "anthropic/claude-opus-5",
            "anthropic/claude-sonnet-5",
        ):
            assert anthropic_uses_adaptive_thinking(model) is True

    def test_bare_claude_ids_are_adaptive(self):
        for model in ("claude-fable-5-1", "claude-fable-5-2", "claude-opus-5", "claude-sonnet-4-6"):
            assert anthropic_uses_adaptive_thinking(model) is True

    def test_opus_and_sonnet_4_6_through_4_8_are_adaptive(self):
        for name in ("claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8"):
            assert anthropic_uses_adaptive_thinking(f"anthropic/{name}") is True

    def test_older_anthropic_models_keep_budget_tokens(self):
        for name in (
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-3-5-sonnet",
        ):
            assert anthropic_uses_adaptive_thinking(f"anthropic/{name}") is False

    def test_non_anthropic_models(self):
        assert anthropic_uses_adaptive_thinking("openai/gpt-6.1-sol") is False
        assert anthropic_uses_adaptive_thinking("google/gemini-3.5-flash") is False


class TestOpenRouterModelId:
    def test_two_part_versions_become_dotted(self):
        assert openrouter_model_id("anthropic/claude-fable-5-1") == "anthropic/claude-fable-5.1"
        assert openrouter_model_id("anthropic/claude-fable-5-2") == "anthropic/claude-fable-5.2"
        assert openrouter_model_id("anthropic/claude-haiku-4-5") == "anthropic/claude-haiku-4.5"
        assert openrouter_model_id("anthropic/claude-opus-4-8") == "anthropic/claude-opus-4.8"

    def test_single_version_ids_are_unchanged(self):
        assert openrouter_model_id("anthropic/claude-opus-5") == "anthropic/claude-opus-5"
        assert openrouter_model_id("anthropic/claude-sonnet-5") == "anthropic/claude-sonnet-5"

    def test_bare_claude_id(self):
        assert openrouter_model_id("claude-fable-5-1") == "claude-fable-5.1"
        assert openrouter_model_id("claude-fable-5-2") == "claude-fable-5.2"

    def test_non_anthropic_ids_are_verbatim(self):
        assert openrouter_model_id("openai/gpt-6.1-sol") == "openai/gpt-6.1-sol"
        assert (
            openrouter_model_id("accounts/fireworks/routers/kimi-k3-us")
            == "accounts/fireworks/routers/kimi-k3-us"
        )


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

    def test_fireworks_kimi_k3_us(self):
        assert model_max_tokens("accounts/fireworks/routers/kimi-k3-us") == 16384

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

    def test_benchmark_covers_critic_route(self):
        assert len(benchmark_models()) == 9
        assert benchmark_models()[-2].model == "google/gemini-3.5-flash"
        assert benchmark_models()[-1].model == "minimax/minimax-m3"

    def test_brainstorm_adds_google_and_minimax_without_changing_council(self):
        assert [entry.model for entry in brainstorm_models()][-2:] == [
            "google/gemini-3.5-flash",
            "minimax/minimax-m3",
        ]
        assert len(brainstorm_models()) == 8

    def test_judge_default(self):
        assert resolved_judge() == "anthropic/claude-fable-5-2"

    def test_gemini_alias_uses_stable_judge(self, monkeypatch):
        monkeypatch.setenv("CONSILIUM_MODEL_JUDGE", "gemini")
        assert resolved_judge() == "google/gemini-3.5-flash"

    def test_critique_default(self):
        assert resolved_critique() == "google/gemini-3.5-flash"

    def test_critique_subscription_fallback(self):
        assert resolved_critique_fallback() == "anthropic/claude-opus-5-5"

    def test_classifier_default(self):
        assert CLASSIFIER_MODEL == "anthropic/claude-opus-5-5"

    def test_judge_subscription_fallback(self):
        assert resolved_judge_fallback() == "openai/gpt-6.1-sol"

    def test_frontier_defaults_preserve_generalist_diversity(self):
        assert [entry.model for entry in resolved_council()] == [
            "openai/gpt-6.1-sol",
            "anthropic/claude-opus-5-5",
            "x-ai/grok-4.7",
            "google/gemini-4-argon",
            "z-ai/glm-5.3",
            "deepseek/deepseek-v4-pro",
        ]

    def test_frontier_default_labels_name_the_served_versions(self):
        assert [entry.name for entry in resolved_council()] == [
            "GPT-6.1-Sol",
            "Claude-Opus-5-5",
            "Grok-4.7",
            "Gemini-4-Argon",
            "GLM-5.3",
            "DeepSeek-V4-Pro",
        ]


class TestSeventhCouncilSeat:
    def test_seventh_seat_is_dormant_by_default(self, monkeypatch):
        monkeypatch.delenv("CONSILIUM_MODEL_M7", raising=False)
        council = resolved_council()
        assert len(council) == 6
        assert [entry.model for entry in council if entry.model.startswith("google/")] == [
            "google/gemini-4-argon"
        ]

        monkeypatch.setenv("CONSILIUM_MODEL_M7", "   ")
        assert len(resolved_council()) == 6

    def test_seventh_seat_override_appends_gemini_entry(self, monkeypatch):
        monkeypatch.setenv("CONSILIUM_MODEL_M7", "google/gemini-4-carbon")
        council = resolved_council()
        assert len(council) == 7
        assert council[-1] == ModelEntry("Gemini-4-Carbon", "google/gemini-4-carbon")

    def test_seventh_seat_joins_quick_brainstorm_and_benchmark_rosters(self, monkeypatch):
        monkeypatch.setenv("CONSILIUM_MODEL_M7", "google/gemini-4-carbon")
        assert len(quick_models()) == 8
        assert len(brainstorm_models()) == 9
        benchmark = [entry.model for entry in benchmark_models()]
        assert benchmark.count("google/gemini-4-carbon") == 1
