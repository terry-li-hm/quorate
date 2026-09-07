"""Model configurations, constants, and shared data structures."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple


class ReasoningEffort(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def anthropic_budget(self) -> int:
        return {"low": 1024, "medium": 8192, "high": 32000}[self.value]

    def google_budget(self) -> int:
        return {"low": 512, "medium": 4096, "high": 16000}[self.value]


class ModelEntry(NamedTuple):
    name: str
    model: str
    fallback: tuple[str, str] | None = None


@dataclass
class ModelCallResult:
    """Result from a single model call with telemetry."""

    name: str
    model_id: str
    response: str
    provider: str = "unknown"
    latency_s: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def is_error(self) -> bool:
        return is_error(self.response)

    def to_dict(self) -> dict:
        result: dict = {
            "model": self.name,
            "model_id": self.model_id,
            "provider": self.provider,
            "latency_s": round(self.latency_s, 2),
        }
        if self.is_error:
            result["error"] = self.response
        else:
            result["response"] = self.response
        if self.tokens_in is not None:
            result["tokens"] = {"input": self.tokens_in, "output": self.tokens_out}
        if self.diagnostics:
            result["diagnostics"] = list(self.diagnostics)
        return result


# API endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
KIMI_CODE_URL = "https://api.kimi.com/coding/v1/chat/completions"
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FIREWORKS_US_K3_MODEL = "accounts/fireworks/routers/kimi-k3-us"
FIREWORKS_US_SERVICE_TIER = "priority"
FIREWORKS_US_CLIENT_CONTRACT = "fireworks-us-k3-priority-v1"
GOOGLE_AI_STUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/models"
ZHIPU_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
XAI_URL = "https://api.x.ai/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Default models
JUDGE_MODEL = "anthropic/claude-fable-5-1"
JUDGE_FALLBACK_MODEL = "openai/gpt-5.6-sol"
CRITIQUE_MODEL = "google/gemini-3.5-flash"
CRITIQUE_FALLBACK_MODEL = "anthropic/claude-opus-5"
CLASSIFIER_MODEL = "anthropic/claude-opus-5"
XAI_DEFAULT_MODEL = "grok-4.5"
BRAINSTORM_EXTRA_MODELS = (
    "google/gemini-3.5-flash",
    "minimax/minimax-m3",
)


def _env(var: str) -> str | None:
    val = os.environ.get(var, "").strip()
    return val if val else None


def _normalize_model(value: str) -> str:
    match value.strip().lower():
        case "sonnet":
            return "anthropic/claude-sonnet-5"
        case "opus":
            return "anthropic/claude-opus-5"
        case "fable":
            return "anthropic/claude-fable-5-1"
        case "gemini":
            return "google/gemini-3.5-flash"
        case _:
            return value.strip()


def resolved_council() -> list[ModelEntry]:
    """Resolve council models at runtime with env overrides."""
    model_1 = _env("CONSILIUM_MODEL_M1") or "openai/gpt-5.6-sol"
    model_2 = _env("CONSILIUM_MODEL_M2") or "anthropic/claude-opus-5"
    model_3 = _env("CONSILIUM_MODEL_M3") or "x-ai/grok-4.5"
    model_4 = _env("CONSILIUM_MODEL_M4") or "kimi-code/k3"
    model_5 = _env("CONSILIUM_MODEL_M5") or "z-ai/glm-5.2"
    model_6 = _env("CONSILIUM_MODEL_M6") or "deepseek/deepseek-v4-pro"
    xai_model = _env("CONSILIUM_XAI_MODEL") or XAI_DEFAULT_MODEL

    return [
        ModelEntry(_display_name(model_1), model_1),
        ModelEntry(_display_name(model_2), model_2),
        ModelEntry(_xai_label(xai_model), model_3),
        ModelEntry(_display_name(model_4), model_4),
        ModelEntry(_display_name(model_5), model_5),
        ModelEntry(_display_name(model_6), model_6),
    ]


def resolved_judge(cli_override: str | None = None) -> str:
    if cli_override:
        return _normalize_model(cli_override)
    return _normalize_model(_env("CONSILIUM_MODEL_JUDGE") or JUDGE_MODEL)


def resolved_judge_fallback() -> str:
    return _normalize_model(_env("CONSILIUM_MODEL_JUDGE_FALLBACK") or JUDGE_FALLBACK_MODEL)


def resolved_critique(cli_override: str | None = None) -> str:
    if cli_override:
        return _normalize_model(cli_override)
    return _normalize_model(_env("CONSILIUM_MODEL_CRITIQUE") or CRITIQUE_MODEL)


def resolved_critique_fallback() -> str:
    return _normalize_model(_env("CONSILIUM_MODEL_CRITIQUE_FALLBACK") or CRITIQUE_FALLBACK_MODEL)


def quick_models() -> list[ModelEntry]:
    judge = resolved_judge()
    judge_label = _display_name(judge)
    models: list[ModelEntry] = [ModelEntry(judge_label, judge, None)]
    models.extend(entry for entry in resolved_council() if entry.model != judge)
    return models


def brainstorm_models() -> list[ModelEntry]:
    """Return the council plus two ideation-only model families."""
    models = list(resolved_council())
    configured = (
        _env("CONSILIUM_BRAINSTORM_MODEL_1") or BRAINSTORM_EXTRA_MODELS[0],
        _env("CONSILIUM_BRAINSTORM_MODEL_2") or BRAINSTORM_EXTRA_MODELS[1],
    )
    for model in configured:
        if all(entry.model != model for entry in models):
            models.append(ModelEntry(_display_name(model), model))
    return models


def benchmark_models() -> list[ModelEntry]:
    """Return every primary production role once for route canaries."""
    models = quick_models()
    production_models = [resolved_critique(), *(entry.model for entry in brainstorm_models())]
    for model in production_models:
        if all(entry.model != model for entry in models):
            models.append(ModelEntry(_display_name(model), model, None))
    return models


def _display_name(model_id: str) -> str:
    name = model_id.rsplit("/", 1)[-1]
    name = name.removesuffix("-preview")
    parts = name.split("-")
    result = []
    for part in parts:
        if not part:
            continue
        match part.lower():
            case "gpt":
                result.append("GPT")
            case "glm":
                result.append("GLM")
            case "deepseek":
                result.append("DeepSeek")
            case "kimi":
                result.append("Kimi")
            case "mimo":
                result.append("MiMo")
            case "minimax":
                result.append("MiniMax")
            case "us":
                result.append("US")
            case _:
                result.append(part[0].upper() + part[1:])
    return "-".join(result)


def _xai_label(model: str) -> str:
    if "4.20" in model:
        suffix = "-NR" if "non-reasoning" in model else ""
        return f"Grok-4.20\u03b2{suffix}"
    if "4.5" in model:
        return "Grok-4.5"
    return _display_name(model)


THINKING_MODELS = {
    "claude-fable-5",
    "claude-fable-5-1",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-5",
    "claude-sonnet-5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4-pro",
    "gpt-5.4",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "grok-4",
    "grok-4.5",
    "grok-4.3",
    "deepseek-r1",
    "deepseek-v4-pro",
    "glm-5",
    "glm-5.2",
    "glm-5.1",
    "kimi-k2.6",
    "kimi-k3-us",
    "k3",
    "mimo-v2.5-pro",
    "minimax-m3",
}


# Anthropic models whose native API takes adaptive thinking (thinking.type
# "adaptive" plus output_config.effort) instead of budget_tokens. Only
# 4.5-and-earlier models still accept budget_tokens.
_ADAPTIVE_THINKING_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)


def anthropic_uses_adaptive_thinking(model: str) -> bool:
    """True for Anthropic models that reject budget_tokens in favor of adaptive thinking."""
    name = model.rsplit("/", 1)[-1].lower()
    return name.startswith(_ADAPTIVE_THINKING_PREFIXES)


def openrouter_model_id(model: str) -> str:
    """Spell an Anthropic id the way OpenRouter lists it (dotted minor version)."""
    if not (model.startswith("anthropic/") or model.startswith("claude")):
        return model
    prefix, _, bare = model.rpartition("/")
    parts = bare.split("-")
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        parts = [*parts[:-2], f"{parts[-2]}.{parts[-1]}"]
    ident = "-".join(parts)
    return f"{prefix}/{ident}" if prefix else ident


def is_thinking_model(model: str) -> bool:
    name = model.rsplit("/", 1)[-1].lower()
    return (
        name in THINKING_MODELS
        or name.startswith("grok-4.2")
        or name.startswith("grok-4.3")
        or name.startswith("grok-4.5")
    )


def model_max_tokens(model: str) -> int:
    lower = model.lower()
    if "gemini-2.5" in lower or "gemini-3" in lower:
        return 65536
    if "gemini" in lower:
        return 8192
    if "claude" in lower or "anthropic" in lower:
        return 32000
    if "gpt" in lower or "openai" in lower or "deepseek" in lower:
        return 16384
    if "grok" in lower or "xai" in lower:
        return 32768
    if "mimo" in lower or "xiaomi" in lower:
        return 32768
    if "kimi" in lower or "moonshot" in lower:
        return 16384
    if "glm" in lower or "zhipu" in lower:
        return 16000
    return 8192


def is_error(content: str) -> bool:
    return not content or (
        content.startswith("[")
        and (
            content.startswith("[Error:")
            or content.startswith("[No response")
            or content.startswith("[Model still thinking")
        )
    )


def api_keys() -> dict[str, str | None]:
    """Load API keys from environment. Keys injected by op run via quorate.env.op."""
    return {
        "openrouter": _env("QUORATE_OPENROUTER_KEY") or _env("OPENROUTER_API_KEY"),
        "kimi_code": _env("KIMI_CODE_API_KEY"),
        "fireworks": _env("QUORATE_FIREWORKS_KEY") or _env("FIREWORKS_API_KEY"),
        "google": _env("GOOGLE_API_KEY"),
        "zhipu": _env("ZHIPU_API_KEY"),
        "xai": _env("XAI_API_KEY"),
        "anthropic": _env("ANTHROPIC_API_KEY"),
        "openai": _env("OPENAI_API_KEY"),
    }


@dataclass
class Message:
    role: str
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls("assistant", content)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}
