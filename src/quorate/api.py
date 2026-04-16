"""HTTP clients — native provider APIs first, OpenRouter fallback."""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from quorate.config import (
    ANTHROPIC_URL,
    ANTHROPIC_VERSION,
    GOOGLE_AI_STUDIO_URL,
    OPENROUTER_URL,
    XAI_URL,
    Message,
    ModelEntry,
    ReasoningEffort,
    api_keys,
    is_error,
    is_thinking_model,
)

THINK_RE = re.compile(r"(?s)<think>.*?</think>")


def _strip_think(content: str) -> str:
    return THINK_RE.sub("", content).strip()


# --- Provider clients ---


async def _openrouter(
    client: httpx.AsyncClient, api_key: str, model: str,
    messages: list[Message], max_tokens: int, timeout: float,
    effort: ReasoningEffort | None,
) -> str:
    if is_thinking_model(model):
        max_tokens = max(max_tokens, 4096)
        timeout = max(timeout, 300)
    body: dict = {"model": model, "messages": [m.to_dict() for m in messages], "max_tokens": max_tokens}
    if effort:
        body["reasoning"] = {"effort": effort.value}
    try:
        resp = await client.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=timeout)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: {model}: {exc}]"
    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from {model}]"
    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]"
    choices = data.get("choices", [])
    content = (choices[0].get("message") or {}).get("content", "").strip() if choices else ""
    return _strip_think(content) if content else f"[No response from {model}]"


async def _anthropic(
    client: httpx.AsyncClient, api_key: str, model: str,
    messages: list[Message], max_tokens: int, timeout: float,
    effort: ReasoningEffort | None,
) -> str:
    bare = model.removeprefix("anthropic/")
    budget = effort.anthropic_budget() if effort else None
    if budget:
        max_tokens = max(max_tokens, budget + 2000)
    body: dict = {"model": bare, "messages": [m.to_dict() for m in messages], "max_tokens": max_tokens}
    if budget:
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    try:
        resp = await client.post(ANTHROPIC_URL, headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}, json=body, timeout=timeout)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: anthropic {bare}: {exc}]"
    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from anthropic {bare}]"
    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]"
    text = next((b["text"] for b in data.get("content", []) if b.get("type") == "text"), "").strip()
    return text if text else f"[No response from anthropic {bare}]"


async def _google(
    client: httpx.AsyncClient, api_key: str, model: str,
    messages: list[Message], max_tokens: int, timeout: float,
    effort: ReasoningEffort | None,
) -> str:
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
        return f"[Error: google {bare}: {exc}]"
    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from google {bare}]"
    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]"
    candidates = data.get("candidates", [])
    parts = (candidates[0].get("content") or {}).get("parts", []) if candidates else []
    text = (parts[0].get("text", "") if parts else "").strip()
    return text if text else f"[No response from google {bare}]"


async def _xai(
    client: httpx.AsyncClient, api_key: str, model: str,
    messages: list[Message], max_tokens: int, timeout: float,
    effort: ReasoningEffort | None,
) -> str:
    body: dict = {"model": model, "messages": [m.to_dict() for m in messages], "max_tokens": max_tokens}
    if effort:
        body["reasoning_effort"] = effort.value
    try:
        resp = await client.post(XAI_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=timeout)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: xai {model}: {exc}]"
    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from xai {model}]"
    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]"
    choices = data.get("choices", [])
    content = (choices[0].get("message") or {}).get("content", "").strip() if choices else ""
    return _strip_think(content) if content else f"[No response from xai {model}]"


async def _openai(
    client: httpx.AsyncClient, api_key: str, model: str,
    messages: list[Message], max_tokens: int, timeout: float,
    effort: ReasoningEffort | None,
) -> str:
    bare = model.removeprefix("openai/")
    body: dict = {"model": bare, "messages": [m.to_dict() for m in messages], "max_tokens": max_tokens}
    try:
        resp = await client.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=timeout)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: openai {bare}: {exc}]"
    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from openai {bare}]"
    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]"
    choices = data.get("choices", [])
    content = (choices[0].get("message") or {}).get("content", "").strip() if choices else ""
    return _strip_think(content) if content else f"[No response from openai {bare}]"


async def _codex_exec(model: str, messages: list[Message], timeout: float) -> str:
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
        return f"[Error: Empty prompt for codex {bare}]"
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "codex", "exec", "-m", bare, prompt,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ), timeout=timeout,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=b"\n"), timeout=timeout
        )
    except (asyncio.TimeoutError, FileNotFoundError) as exc:
        return f"[Error: codex exec {bare}: {exc}]"
    if proc.returncode != 0:
        err = (stderr or b"").decode().strip()
        return f"[Error: codex exec {bare}: exit {proc.returncode} {err[:100]}]"
    # In non-TTY mode, stdout contains just the response text
    result = stdout.decode().strip()
    return result if result else f"[No response from codex {bare}]"


async def _claude_print(model: str, messages: list[Message], timeout: float) -> str:
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
        return f"[Error: Empty prompt for {bare}]"
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "claude", "--model", bare, "--print", "--output-format", "json", "-p", prompt,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            ), timeout=timeout,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, FileNotFoundError) as exc:
        return f"[Error: claude --print {bare}: {exc}]"
    if proc.returncode != 0:
        return f"[Error: claude --print {bare}: {(stderr or b'').decode().strip() or f'exit {proc.returncode}'}]"
    try:
        data = json.loads(stdout.decode())
    except json.JSONDecodeError:
        return f"[Error: Invalid JSON from claude --print {bare}]"
    if data.get("is_error"):
        return f"[Error: {data.get('result', 'Unknown')}]"
    result = (data.get("result") or "").strip()
    if not result or "Credit balance is too low" in result:
        return f"[Error: {result or 'empty'}]"
    return _strip_think(result)


# --- Routing ---


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
    return "openrouter"


async def query_model(
    client: httpx.AsyncClient,
    keys: dict[str, str | None],
    entry: ModelEntry,
    messages: list[Message],
    max_tokens: int = 4096,
    timeout: float = 90,
    effort: ReasoningEffort | None = None,
) -> tuple[str, str, str]:
    """Query model: native provider first, OpenRouter fallback.
    Returns (name, model_used, response).
    """
    model_name = entry.model.rsplit("/", 1)[-1]
    provider = _detect_provider(entry.model)

    # Try native provider
    if provider == "anthropic":
        # claude --print first (Max sub)
        response = await _claude_print(entry.model, messages, timeout)
        if not is_error(response):
            return (entry.name, model_name, response)
        # Then Anthropic API
        if keys.get("anthropic"):
            response = await _anthropic(client, keys["anthropic"], entry.model, messages, max_tokens, timeout, effort)
            if not is_error(response):
                return (entry.name, model_name, response)

    elif provider == "google" and keys.get("google"):
        response = await _google(client, keys["google"], entry.model, messages, max_tokens, timeout, effort)
        if not is_error(response):
            return (entry.name, model_name, response)

    elif provider == "xai" and keys.get("xai"):
        response = await _xai(client, keys["xai"], entry.model, messages, max_tokens, timeout, effort)
        if not is_error(response):
            return (entry.name, model_name, response)

    elif provider == "openai":
        # Codex CLI first (Pro subscription, free)
        response = await _codex_exec(entry.model, messages, timeout)
        if not is_error(response):
            return (entry.name, model_name, response)
        # Then native OpenAI API
        if keys.get("openai"):
            response = await _openai(client, keys["openai"], entry.model, messages, max_tokens, timeout, effort)
            if not is_error(response):
                return (entry.name, model_name, response)

    # OpenRouter fallback
    if keys.get("openrouter"):
        response = await _openrouter(client, keys["openrouter"], entry.model, messages, max_tokens, timeout, effort)
        if not is_error(response):
            return (entry.name, model_name, response)

    return (entry.name, model_name, f"[Error: All providers failed for {entry.name}]")


async def run_parallel(
    models: list[ModelEntry],
    messages: list[Message],
    max_tokens: int = 4096,
    timeout: float = 90,
    effort: ReasoningEffort | None = None,
) -> list[tuple[str, str, str]]:
    """Query all models in parallel."""
    keys = api_keys()
    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.wait_for(
                query_model(client, keys, entry, messages, max_tokens, timeout, effort),
                timeout=max(timeout, 180 if is_thinking_model(entry.model) else timeout),
            )
            for entry in models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    output = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            output.append((models[idx].name, models[idx].name, f"[Error: {models[idx].name} timed out]"))
        else:
            output.append(result)
    return output


async def query_judge(
    model: str, messages: list[Message],
    max_tokens: int = 16384, timeout: float = 300,
    effort: ReasoningEffort | None = None,
) -> str:
    """Query judge model with native-first fallback."""
    keys = api_keys()
    provider = _detect_provider(model)
    async with httpx.AsyncClient() as client:
        if provider == "google" and keys.get("google"):
            response = await _google(client, keys["google"], model, messages, max_tokens, timeout, effort)
            if not is_error(response):
                return response
        if provider == "anthropic":
            response = await _claude_print(model, messages, timeout)
            if not is_error(response):
                return response
            if keys.get("anthropic"):
                response = await _anthropic(client, keys["anthropic"], model, messages, max_tokens, timeout, effort)
                if not is_error(response):
                    return response
        if keys.get("openrouter"):
            return await _openrouter(client, keys["openrouter"], model, messages, max_tokens, timeout, effort)
    return f"[Error: No providers available for judge {model}]"
