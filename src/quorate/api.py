"""HTTP clients — native provider APIs first, OpenRouter fallback."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx

from quorate.config import (
    ANTHROPIC_URL,
    ANTHROPIC_VERSION,
    GOOGLE_AI_STUDIO_URL,
    OPENROUTER_URL,
    XAI_URL,
    ZHIPU_URL,
    Message,
    ModelCallResult,
    ModelEntry,
    ReasoningEffort,
    api_keys,
    is_error,
    is_thinking_model,
)

# Type alias for provider function returns: (content, {tokens_in, tokens_out} | None)
type ProviderResult = tuple[str, dict[str, int] | None]

THINK_RE = re.compile(r"(?s)<think>.*?</think>")


def _strip_think(content: str) -> str:
    return THINK_RE.sub("", content).strip()


def _diagnostic_code(content: str) -> str:
    """Reduce a provider error to a safe, non-sensitive diagnostic code."""
    lower = content.lower()
    http_status = re.search(r"http\s+(\d{3})", lower)
    if http_status:
        return f"http_{http_status.group(1)}"
    if "credit balance" in lower or "insufficient credit" in lower:
        return "credits"
    if "timed out" in lower or "timeout" in lower:
        return "timeout"
    if "invalid json" in lower:
        return "invalid_json"
    if "no response" in lower or "empty" in lower:
        return "empty_response"
    if "not found" in lower or "no such file" in lower:
        return "unavailable"
    return "provider_error"


# --- Provider clients ---


async def _openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[Message],
    max_tokens: int,
    timeout: float,
    effort: ReasoningEffort | None,
) -> ProviderResult:
    if is_thinking_model(model):
        max_tokens = max(max_tokens, 4096)
    body: dict = {
        "model": model,
        "messages": [m.to_dict() for m in messages],
        "max_tokens": max_tokens,
    }
    if effort:
        body["reasoning"] = {"effort": effort.value}
    try:
        resp = await client.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=timeout,
        )
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: {model}: {exc}]", None
    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from {model}]", None
    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]", None
    usage = data.get("usage")
    tokens = (
        {"tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens")}
        if usage
        else None
    )
    choices = data.get("choices", [])
    content = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    text = _strip_think(content) if content else f"[No response from {model}]"
    return text, tokens


async def _anthropic(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[Message],
    max_tokens: int,
    timeout: float,
    effort: ReasoningEffort | None,
) -> ProviderResult:
    bare = model.removeprefix("anthropic/")
    budget = effort.anthropic_budget() if effort else None
    if budget:
        max_tokens = max(max_tokens, budget + 2000)
    body: dict = {
        "model": bare,
        "messages": [m.to_dict() for m in messages],
        "max_tokens": max_tokens,
    }
    if budget:
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    try:
        resp = await client.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=body,
            timeout=timeout,
        )
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: anthropic {bare}: {exc}]", None
    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from anthropic {bare}]", None
    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]", None
    usage = data.get("usage")
    tokens = (
        {"tokens_in": usage.get("input_tokens"), "tokens_out": usage.get("output_tokens")}
        if usage
        else None
    )
    text = next((b["text"] for b in data.get("content", []) if b.get("type") == "text"), "").strip()
    return (text if text else f"[No response from anthropic {bare}]"), tokens


async def _google(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[Message],
    max_tokens: int,
    timeout: float,
    effort: ReasoningEffort | None,
) -> ProviderResult:
    bare = model.removeprefix("google/")
    contents, sys_text = [], None
    for msg in messages:
        if msg.role == "system":
            sys_text = msg.content
        elif msg.role == "user":
            contents.append({"role": "user", "parts": [{"text": msg.content}]})
        elif msg.role == "assistant":
            contents.append({"role": "model", "parts": [{"text": msg.content}]})
    gen_config: dict = {"maxOutputTokens": max_tokens}
    if effort:
        gen_config["thinkingConfig"] = {"thinkingBudget": effort.google_budget()}
    body: dict = {"contents": contents, "generationConfig": gen_config}
    if sys_text:
        body["systemInstruction"] = {"parts": [{"text": sys_text}]}
    url = f"{GOOGLE_AI_STUDIO_URL}/{bare}:generateContent?key={api_key}"
    try:
        resp = await client.post(url, json=body, timeout=timeout)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: google {bare}: {exc}]", None
    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from google {bare}]", None
    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]", None
    usage = data.get("usageMetadata")
    tokens = (
        {
            "tokens_in": usage.get("promptTokenCount"),
            "tokens_out": usage.get("candidatesTokenCount"),
        }
        if usage
        else None
    )
    candidates = data.get("candidates", [])
    parts = (candidates[0].get("content") or {}).get("parts", []) if candidates else []
    text = (parts[0].get("text", "") if parts else "").strip()
    return (text if text else f"[No response from google {bare}]"), tokens


async def _xai(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[Message],
    max_tokens: int,
    timeout: float,
    effort: ReasoningEffort | None,
) -> ProviderResult:
    bare = model.removeprefix("x-ai/")
    body: dict = {
        "model": bare,
        "messages": [m.to_dict() for m in messages],
        "max_tokens": max_tokens,
    }
    if effort:
        body["reasoning_effort"] = effort.value
    try:
        resp = await client.post(
            XAI_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=timeout
        )
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: xai {model}: {exc}]", None
    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from xai {model}]", None
    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]", None
    usage = data.get("usage")
    tokens = (
        {"tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens")}
        if usage
        else None
    )
    choices = data.get("choices", [])
    content = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    text = _strip_think(content) if content else f"[No response from xai {model}]"
    return text, tokens


async def _openai(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[Message],
    max_tokens: int,
    timeout: float,
    effort: ReasoningEffort | None,
) -> ProviderResult:
    bare = model.removeprefix("openai/")
    body: dict = {
        "model": bare,
        "messages": [m.to_dict() for m in messages],
        "max_tokens": max_tokens,
    }
    if effort:
        body["reasoning_effort"] = effort.value if hasattr(effort, "value") else str(effort)
    try:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=timeout,
        )
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: openai {bare}: {exc}]", None
    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from openai {bare}]", None
    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]", None
    usage = data.get("usage")
    tokens = (
        {"tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens")}
        if usage
        else None
    )
    choices = data.get("choices", [])
    content = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    text = _strip_think(content) if content else f"[No response from openai {bare}]"
    return text, tokens


async def _codex_exec(
    model: str,
    messages: list[Message],
    timeout: float,
    effort: ReasoningEffort | None,
) -> ProviderResult:
    """Query Codex CLI exec mode (uses Codex Pro subscription)."""
    bare = model.removeprefix("openai/")
    # Codex uses base model names (gpt-5.4, not gpt-5.4-pro)
    if bare.endswith("-pro"):
        bare = bare.removesuffix("-pro")
    # Combine messages into a single prompt
    sections = []
    for msg in messages:
        text = msg.content.strip()
        if not text:
            continue
        sections.append(f"Previous response:\n{text}" if msg.role == "assistant" else text)
    prompt = "\n\n".join(sections)
    if not prompt:
        return f"[Error: Empty prompt for codex {bare}]", None
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        outfile = tmp.name
    reasoning_effort = effort.value if effort else "xhigh"
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "codex",
                "exec",
                "-m",
                bare,
                "-o",
                outfile,
                "--skip-git-repo-check",
                "-c",
                f'model_reasoning_effort="{reasoning_effort}"',
                prompt,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=timeout,
        )
        await asyncio.wait_for(proc.communicate(input=b"\n"), timeout=timeout)
    except (asyncio.TimeoutError, FileNotFoundError) as exc:
        return f"[Error: codex exec {bare}: {exc}]", None
    finally:
        import os

        result = ""
        if os.path.exists(outfile):
            result = Path(outfile).read_text().strip()
            os.unlink(outfile)
    if proc.returncode != 0:
        return f"[Error: codex exec {bare}: exit {proc.returncode}]", None
    return (result if result else f"[No response from codex {bare}]"), None


async def _claude_print(
    model: str,
    messages: list[Message],
    timeout: float,
    effort: ReasoningEffort | None,
) -> ProviderResult:
    """Query Claude Code CLI --print (Max subscription)."""
    bare = model.removeprefix("anthropic/")
    sections = []
    for msg in messages:
        text = msg.content.strip()
        if not text:
            continue
        sections.append(f"Previous response:\n{text}" if msg.role == "assistant" else text)
    prompt = "\n\n".join(sections)
    if not prompt:
        return f"[Error: Empty prompt for {bare}]", None
    # Strip ANTHROPIC_API_KEY so claude --print uses Max subscription, not depleted API credits
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    proc = None
    command = ["claude", "--model", bare, "--print", "--output-format", "json"]
    if effort:
        command.extend(["--effort", effort.value])
    command.extend(["-p", prompt])
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            ),
            timeout=timeout,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, FileNotFoundError) as exc:
        if proc is not None:
            proc.kill()
        return f"[Error: claude --print {bare}: {exc}]", None
    if proc.returncode != 0:
        detail = (stderr or b"").decode().strip() or f"exit {proc.returncode}"
        return (
            f"[Error: claude --print {bare}: {detail}]",
            None,
        )
    try:
        data = json.loads(stdout.decode())
    except json.JSONDecodeError:
        return f"[Error: Invalid JSON from claude --print {bare}]", None
    if data.get("is_error"):
        return f"[Error: {data.get('result', 'Unknown')}]", None
    result = (data.get("result") or "").strip()
    if not result or "Credit balance is too low" in result:
        return f"[Error: {result or 'empty'}]", None
    # claude --print JSON includes token usage
    usage = data.get("usage")
    tokens = (
        {"tokens_in": usage.get("input_tokens"), "tokens_out": usage.get("output_tokens")}
        if usage
        else None
    )
    return _strip_think(result), tokens


def _antigravity_model(model: str, effort: ReasoningEffort | None) -> str:
    """Map a Quorate Google model to an Antigravity model label."""
    bare = model.removeprefix("google/")
    level = (effort or ReasoningEffort.MEDIUM).value.title()
    if "gemini-3.1-pro" in bare:
        level = "Low" if level == "Low" else "High"
        return f"Gemini 3.1 Pro ({level})"
    return f"Gemini 3.5 Flash ({level})"


async def _antigravity_prompt(
    model: str,
    messages: list[Message],
    timeout: float,
    effort: ReasoningEffort | None,
) -> ProviderResult:
    """Query Gemini through the Antigravity subscription CLI."""
    label = _antigravity_model(model, effort)
    sections = []
    for msg in messages:
        text = msg.content.strip()
        if not text:
            continue
        sections.append(f"Previous response:\n{text}" if msg.role == "assistant" else text)
    prompt = "\n\n".join(sections)
    if not prompt:
        return f"[Error: Empty prompt for Antigravity {label}]", None
    proc = None
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "agy",
                "--print",
                prompt,
                "--sandbox",
                "--mode",
                "plan",
                "--model",
                label,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=timeout,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, FileNotFoundError) as exc:
        if proc is not None:
            proc.kill()
        return f"[Error: agy --print {label}: {exc}]", None
    if proc.returncode != 0:
        detail = (stderr or b"").decode().strip() or f"exit {proc.returncode}"
        return (
            f"[Error: agy --print {label}: {detail}]",
            None,
        )
    result = stdout.decode().strip()
    if not result:
        return f"[No response from Antigravity {label}]", None
    return _strip_think(result), None


async def _zhipu(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[Message],
    max_tokens: int,
    timeout: float,
    effort: ReasoningEffort | None,
) -> ProviderResult:
    """Query ZhiPu API directly (free tier)."""
    bare = model.removeprefix("z-ai/").removeprefix("zhipu/")
    body: dict = {
        "model": bare,
        "messages": [m.to_dict() for m in messages],
        "max_tokens": max_tokens,
    }
    try:
        resp = await client.post(
            ZHIPU_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=timeout,
        )
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: zhipu {bare}: {exc}]", None
    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from zhipu {bare}]", None
    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]", None
    usage = data.get("usage")
    tokens = (
        {"tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens")}
        if usage
        else None
    )
    choices = data.get("choices", [])
    content = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    text = _strip_think(content) if content else f"[No response from zhipu {model}]"
    return text, tokens


def _detect_provider(model: str) -> str:
    """Detect native provider from model ID."""
    if "anthropic/" in model or "claude" in model:
        return "anthropic"
    if "google/" in model or "gemini" in model:
        return "google"
    if "x-ai/" in model or "grok" in model:
        return "xai"
    if "openai/" in model or "gpt" in model:
        return "openai"
    if "glm" in model or "zhipu" in model:
        return "zhipu"
    return "openrouter"


async def query_model(
    client: httpx.AsyncClient,
    keys: dict[str, str | None],
    entry: ModelEntry,
    messages: list[Message],
    max_tokens: int = 4096,
    timeout: float = 90,
    effort: ReasoningEffort | None = None,
) -> ModelCallResult:
    """Query model: native provider first, OpenRouter fallback.
    Returns ModelCallResult with telemetry.
    """
    model_name = entry.model.rsplit("/", 1)[-1]
    provider = _detect_provider(entry.model)
    if is_thinking_model(entry.model):
        timeout = max(timeout, 180)
    start = time.monotonic()
    deadline = start + timeout
    subscription_timeout = min(120.0, timeout * 0.75)
    diagnostics: list[str] = []

    def _remaining_timeout() -> float:
        return max(1.0, deadline - time.monotonic())

    def _record_failure(route: str, content: str) -> None:
        diagnostics.append(f"{route}:{_diagnostic_code(content)}")

    def _record_missing(route: str) -> None:
        diagnostics.append(f"{route}:no_credentials")

    def _result(
        content: str,
        used_provider: str,
        tokens: dict[str, int] | None = None,
    ) -> ModelCallResult:
        return ModelCallResult(
            name=entry.name,
            model_id=model_name,
            response=content,
            provider=used_provider,
            latency_s=time.monotonic() - start,
            tokens_in=tokens.get("tokens_in") if tokens else None,
            tokens_out=tokens.get("tokens_out") if tokens else None,
            diagnostics=tuple(diagnostics),
        )

    # Try native provider
    if provider == "anthropic":
        content, tokens = await _claude_print(
            entry.model,
            messages,
            min(subscription_timeout, _remaining_timeout()),
            effort,
        )
        if not is_error(content):
            return _result(content, "claude-print", tokens)
        _record_failure("claude-print", content)
        anthropic_key = keys.get("anthropic")
        if anthropic_key:
            content, tokens = await _anthropic(
                client,
                anthropic_key,
                entry.model,
                messages,
                max_tokens,
                _remaining_timeout(),
                effort,
            )
            if not is_error(content):
                return _result(content, "anthropic-api", tokens)
            _record_failure("anthropic-api", content)
        else:
            _record_missing("anthropic-api")

    elif provider == "google":
        content, tokens = await _antigravity_prompt(
            entry.model,
            messages,
            min(subscription_timeout, _remaining_timeout()),
            effort,
        )
        if not is_error(content):
            return _result(content, "antigravity-cli", tokens)
        _record_failure("antigravity-cli", content)
        google_key = keys.get("google")
        if google_key:
            content, tokens = await _google(
                client,
                google_key,
                entry.model,
                messages,
                max_tokens,
                _remaining_timeout(),
                effort,
            )
            if not is_error(content):
                return _result(content, "google-ai-studio", tokens)
            _record_failure("google-ai-studio", content)
        else:
            _record_missing("google-ai-studio")

    elif provider == "xai":
        xai_key = keys.get("xai")
        if xai_key:
            content, tokens = await _xai(
                client,
                xai_key,
                entry.model,
                messages,
                max_tokens,
                _remaining_timeout(),
                effort,
            )
            if not is_error(content):
                return _result(content, "xai-native", tokens)
            _record_failure("xai-native", content)
        else:
            _record_missing("xai-native")

    elif provider == "openai":
        content, tokens = await _codex_exec(
            entry.model,
            messages,
            min(subscription_timeout, _remaining_timeout()),
            effort,
        )
        if not is_error(content):
            return _result(content, "codex-exec", tokens)
        _record_failure("codex-exec", content)
        openai_key = keys.get("openai")
        if openai_key:
            content, tokens = await _openai(
                client,
                openai_key,
                entry.model,
                messages,
                max_tokens,
                _remaining_timeout(),
                effort,
            )
            if not is_error(content):
                return _result(content, "openai-api", tokens)
            _record_failure("openai-api", content)
        else:
            _record_missing("openai-api")

    elif provider == "zhipu":
        zhipu_key = keys.get("zhipu")
        if zhipu_key:
            content, tokens = await _zhipu(
                client,
                zhipu_key,
                entry.model,
                messages,
                max_tokens,
                _remaining_timeout(),
                effort,
            )
            if not is_error(content):
                return _result(content, "zhipu-native", tokens)
            _record_failure("zhipu-native", content)
        else:
            _record_missing("zhipu-native")

    # OpenRouter fallback
    openrouter_key = keys.get("openrouter")
    if openrouter_key:
        content, tokens = await _openrouter(
            client,
            openrouter_key,
            entry.model,
            messages,
            max_tokens,
            _remaining_timeout(),
            effort,
        )
        if not is_error(content):
            return _result(content, "openrouter", tokens)
        _record_failure("openrouter", content)
    else:
        _record_missing("openrouter")

    return _result(f"[Error: All providers failed for {entry.name}]", "none")


async def run_parallel(
    models: list[ModelEntry],
    messages: list[Message],
    max_tokens: int = 4096,
    timeout: float = 90,
    effort: ReasoningEffort | None = None,
) -> list[ModelCallResult]:
    """Query all models in parallel."""
    keys = api_keys()
    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.wait_for(
                query_model(client, keys, entry, messages, max_tokens, timeout, effort),
                timeout=max(timeout, 180 if is_thinking_model(entry.model) else timeout) + 5,
            )
            for entry in models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    output: list[ModelCallResult] = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            output.append(
                ModelCallResult(
                    name=models[idx].name,
                    model_id=models[idx].name,
                    response=f"[Error: {models[idx].name} timed out]",
                    provider="timeout",
                    diagnostics=("orchestrator:timeout",),
                )
            )
        else:
            output.append(result)
    return output


def quorum_health(results: list[ModelCallResult]) -> tuple[int, int, bool]:
    """Return successful seats, strict-majority target, and quorum state."""
    success_count = sum(not result.is_error for result in results)
    quorum_target = max(2, len(results) // 2 + 1)
    return success_count, quorum_target, success_count >= quorum_target


async def query_judge(
    model: str,
    messages: list[Message],
    max_tokens: int = 16384,
    timeout: float = 300,
    effort: ReasoningEffort | None = None,
) -> str:
    """Query judge model with native-first fallback. Returns content string."""
    keys = api_keys()
    provider = _detect_provider(model)
    async with httpx.AsyncClient() as client:
        if provider == "google":
            content, _ = await _antigravity_prompt(model, messages, timeout, effort)
            if not is_error(content):
                return content
            google_key = keys.get("google")
            if google_key:
                content, _ = await _google(
                    client, google_key, model, messages, max_tokens, timeout, effort
                )
                if not is_error(content):
                    return content
        if provider == "anthropic":
            content, _ = await _claude_print(model, messages, timeout, effort)
            if not is_error(content):
                return content
            anthropic_key = keys.get("anthropic")
            if anthropic_key:
                content, _ = await _anthropic(
                    client, anthropic_key, model, messages, max_tokens, timeout, effort
                )
                if not is_error(content):
                    return content
        if provider == "openai":
            content, _ = await _codex_exec(model, messages, timeout, effort)
            if not is_error(content):
                return content
            openai_key = keys.get("openai")
            if openai_key:
                content, _ = await _openai(
                    client, openai_key, model, messages, max_tokens, timeout, effort
                )
                if not is_error(content):
                    return content
        if provider == "zhipu":
            zhipu_key = keys.get("zhipu")
            if zhipu_key:
                content, _ = await _zhipu(
                    client, zhipu_key, model, messages, max_tokens, timeout, effort
                )
                if not is_error(content):
                    return content
        openrouter_key = keys.get("openrouter")
        if openrouter_key:
            content, _ = await _openrouter(
                client, openrouter_key, model, messages, max_tokens, timeout, effort
            )
            return content
    return f"[Error: No providers available for judge {model}]"
