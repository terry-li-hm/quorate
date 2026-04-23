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
        return result


# API endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_AI_STUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/models"
ZHIPU_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
XAI_URL = "https://api.x.ai/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Default models
JUDGE_MODEL = "google/gemini-3.1-pro-preview"
CRITIQUE_MODEL = "anthropic/claude-opus-4-6"
CLASSIFIER_MODEL = "anthropic/claude-opus-4-6"
XAI_DEFAULT_MODEL = "grok-4.20-0309-reasoning"


def _env(var: str) -> str | None:
    val = os.environ.get(var, "").strip()
    return val if val else None


def _normalize_model(value: str) -> str:
    match value.strip().lower():
        case "sonnet":
            return "anthropic/claude-sonnet-4-6"
        case "opus":
            return "anthropic/claude-opus-4-6"
        case "gemini":
            return "google/gemini-3.1-pro-preview"
        case _:
            return value.strip()


def resolved_council() -> list[ModelEntry]:
    """Resolve council models at runtime with env overrides."""
    model_1 = _env("CONSILIUM_MODEL_M1") or "openai/gpt-5.4-pro"
    model_2 = _env("CONSILIUM_MODEL_M2") or "anthropic/claude-opus-4-6"
    model_3 = _env("CONSILIUM_MODEL_M3") or "x-ai/grok-4.20-0309-reasoning"
    model_4 = _env("CONSILIUM_MODEL_M4") or "qwen/qwen3.6-plus"
    model_5 = _env("CONSILIUM_MODEL_M5") or "glm-5.1"
    xai_model = _env("CONSILIUM_XAI_MODEL") or XAI_DEFAULT_MODEL

    return [
        ModelEntry(_display_name(model_1), model_1),
        ModelEntry(_display_name(model_2), model_2),
        ModelEntry(_xai_label(xai_model), model_3),
        ModelEntry(_display_name(model_4), model_4),
        ModelEntry(_display_name(model_5), model_5),
    ]


def resolved_judge(cli_override: str | None = None) -> str:
    if cli_override:
        return _normalize_model(cli_override)
    return _normalize_model(_env("CONSILIUM_MODEL_JUDGE") or JUDGE_MODEL)


def resolved_critique(cli_override: str | None = None) -> str:
    if cli_override:
        return _normalize_model(cli_override)
    return _normalize_model(_env("CONSILIUM_MODEL_CRITIQUE") or CRITIQUE_MODEL)


def quick_models() -> list[ModelEntry]:
    judge = resolved_judge()
    judge_label = _display_name(judge)
    models: list[ModelEntry] = [ModelEntry(judge_label, judge, None)]
    models.extend(entry for entry in resolved_council() if entry.model != judge)
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
            case _:
                result.append(part[0].upper() + part[1:])
    return "-".join(result)


def _xai_label(model: str) -> str:
    if "4.20" in model:
        suffix = "-NR" if "non-reasoning" in model else ""
        return f"Grok-4.20\u03b2{suffix}"
    return _display_name(model)


THINKING_MODELS = {
    "claude-opus-4-6", "claude-opus-4.5", "gpt-5.4-pro", "gpt-5.4", "gpt-5.2-pro", "gpt-5.2",
    "gemini-3.1-pro-preview", "grok-4", "deepseek-r1", "glm-5", "glm-5.1",
}


def is_thinking_model(model: str) -> bool:
    name = model.rsplit("/", 1)[-1].lower()
    return name in THINKING_MODELS or name.startswith("grok-4.2")


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


def _op_read(item: str) -> str | None:
    """Try reading a credential from 1Password Agents vault."""
    import subprocess
    try:
        result = subprocess.run(
            ["op", "item", "get", item, "--vault=Agents", "--fields=credential", "--reveal"],
            capture_output=True, text=True, timeout=15,
        )
        val = result.stdout.strip()
        return val if val else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# 1Password item names for quorate keys
_OP_ITEMS = {
    "openrouter": "OpenRouter API Key (quorate)",
    "google": "Google AI Studio Key (quorate)",
    "xai": "xAI API Key (quorate)",
    "zhipu": "zhipu-api-key",
}


def api_keys() -> dict[str, str | None]:
    """Load API keys from environment, falling back to 1Password."""
    keys = {
        "openrouter": _env("QUORATE_OPENROUTER_KEY") or _env("OPENROUTER_API_KEY"),
        "google": _env("GOOGLE_API_KEY"),
        "zhipu": _env("ZHIPU_API_KEY"),
        "xai": _env("XAI_API_KEY"),
        "anthropic": _env("ANTHROPIC_API_KEY"),
        "openai": _env("OPENAI_API_KEY"),
    }
    # Fill missing keys from 1Password
    for key_name, op_item in _OP_ITEMS.items():
        if not keys.get(key_name):
            keys[key_name] = _op_read(op_item)
    return keys


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
