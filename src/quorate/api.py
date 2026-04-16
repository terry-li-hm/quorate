"""HTTP clients, parallel queries, retry, and fallback."""

from __future__ import annotations

import asyncio
import random
import re
import subprocess
import time

import httpx

from quorate.config import (
    ANTHROPIC_URL,
    ANTHROPIC_VERSION,
    BIGMODEL_URL,
    GOOGLE_AI_STUDIO_URL,
    OPENROUTER_URL,
    XAI_URL,
    Message,
    ModelEntry,
    ReasoningEffort,
    api_keys,
    is_error,
    is_thinking_model,
    model_max_tokens,
)

THINK_RE = re.compile(r"(?s)<think>.*?</think>")


def _strip_think(content: str) -> str:
    return THINK_RE.sub("", content).strip()


async def _retry_with_backoff(
    func,
    retries: int = 2,
) -> str:
    """Call func() with exponential backoff on failure."""
    for attempt in range(retries + 1):
        if attempt > 0:
            await asyncio.sleep(2**attempt + random.random())
        result = await func()
        if not is_error(result) or attempt == retries:
            return result
    return result


# --- Provider-specific clients ---


async def query_openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[Message],
    max_tokens: int = 4096,
    timeout: float = 120,
    effort: ReasoningEffort | None = None,
) -> str:
    """Query via OpenRouter."""
    if is_thinking_model(model):
        max_tokens = max(max_tokens, 4096)
        timeout = max(timeout, 300)

    body: dict = {
        "model": model,
        "messages": [msg.to_dict() for msg in messages],
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
        return f"[Error: Connection failed for {model}: {exc}]"

    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from {model}]"

    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]"

    choices = data.get("choices", [])
    if not choices:
        return f"[Error: No response from {model}]"

    content = (choices[0].get("message") or {}).get("content", "").strip()
    if not content:
        return f"[No response from {model}]"
    return _strip_think(content)


async def query_anthropic(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[Message],
    max_tokens: int = 4096,
    timeout: float = 120,
    effort: ReasoningEffort | None = None,
) -> str:
    """Query Anthropic Messages API directly."""
    bare = model.removeprefix("anthropic/")
    thinking_budget = effort.anthropic_budget() if effort else None
    if thinking_budget:
        max_tokens = max(max_tokens, thinking_budget + 2000)

    body: dict = {
        "model": bare,
        "messages": [msg.to_dict() for msg in messages],
        "max_tokens": max_tokens,
    }
    if thinking_budget:
        body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

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
        return f"[Error: Connection failed for anthropic {bare}: {exc}]"

    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from anthropic {bare}]"

    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]"

    content_blocks = data.get("content", [])
    text = next(
        (block["text"] for block in content_blocks if block.get("type") == "text"),
        "",
    ).strip()
    return text if text else f"[No response from anthropic {bare}]"


async def query_claude_print(
    model: str,
    messages: list[Message],
    timeout: float = 120,
) -> str:
    """Query Claude Code CLI in print mode (uses Max subscription via OAuth)."""
    bare = model.removeprefix("anthropic/")
    sections = []
    for msg in messages:
        content = msg.content.strip()
        if not content:
            continue
        if msg.role == "assistant":
            sections.append(f"Previous response:\n{content}")
        else:
            sections.append(content)
    prompt = "\n\n".join(sections)
    if not prompt:
        return f"[Error: Empty prompt for claude --print {bare}]"

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "claude", "--model", bare, "--print", "--output-format", "json", "-p", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=timeout,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, FileNotFoundError) as exc:
        return f"[Error: claude --print failed for {bare}: {exc}]"

    if proc.returncode != 0:
        err = stderr.decode().strip() if stderr else f"exit {proc.returncode}"
        return f"[Error: claude --print failed for {bare}: {err}]"

    import json
    try:
        data = json.loads(stdout.decode())
    except json.JSONDecodeError as exc:
        return f"[Error: Invalid JSON from claude --print for {bare}: {exc}]"

    if data.get("is_error"):
        return f"[Error: {data.get('result', 'Unknown')}]"

    result = (data.get("result") or "").strip()
    if not result:
        return f"[Error: Empty result from claude --print for {bare}]"
    if "Credit balance is too low" in result:
        return f"[Error: {result}]"
    return _strip_think(result)


async def query_google(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[Message],
    max_tokens: int = 4096,
    timeout: float = 120,
    effort: ReasoningEffort | None = None,
) -> str:
    """Query Google AI Studio directly."""
    bare = model.removeprefix("google/")
    contents = []
    system_text = None
    for msg in messages:
        if msg.role == "system":
            system_text = msg.content
        elif msg.role == "user":
            contents.append({"role": "user", "parts": [{"text": msg.content}]})
        elif msg.role == "assistant":
            contents.append({"role": "model", "parts": [{"text": msg.content}]})

    gen_config: dict = {"maxOutputTokens": max_tokens}
    if effort:
        gen_config["thinkingConfig"] = {"thinkingBudget": effort.google_budget()}

    body: dict = {"contents": contents, "generationConfig": gen_config}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}

    url = f"{GOOGLE_AI_STUDIO_URL}/{bare}:generateContent?key={api_key}"
    try:
        resp = await client.post(url, json=body, timeout=timeout)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: Request failed for AI Studio {bare}: {exc}]"

    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from AI Studio {bare}]"

    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]"

    candidates = data.get("candidates", [])
    if not candidates:
        return f"[Error: No candidates from AI Studio {bare}]"

    parts = (candidates[0].get("content") or {}).get("parts", [])
    text = (parts[0].get("text", "") if parts else "").strip()
    return text if text else f"[No response from AI Studio {bare}]"


async def query_bigmodel(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[Message],
    max_tokens: int = 4096,
    timeout: float = 120,
) -> str:
    """Query ZhiPu bigmodel.cn directly."""
    body = {
        "model": model,
        "messages": [msg.to_dict() for msg in messages],
        "max_tokens": max_tokens,
    }
    try:
        resp = await client.post(
            BIGMODEL_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=timeout,
        )
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: Connection failed for bigmodel {model}: {exc}]"

    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from bigmodel {model}]"

    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]"

    choices = data.get("choices", [])
    if not choices:
        return f"[Error: No response from bigmodel {model}]"

    content = (choices[0].get("message") or {}).get("content", "").strip()
    if not content:
        return f"[No response from bigmodel {model}]"
    return _strip_think(content)


async def query_xai(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[Message],
    max_tokens: int = 4096,
    timeout: float = 120,
    effort: ReasoningEffort | None = None,
) -> str:
    """Query xAI directly."""
    body: dict = {
        "model": model,
        "messages": [msg.to_dict() for msg in messages],
        "max_tokens": max_tokens,
    }
    if effort:
        body["reasoning_effort"] = effort.value

    try:
        resp = await client.post(
            XAI_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=timeout,
        )
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return f"[Error: Connection failed for xai {model}: {exc}]"

    if resp.status_code != 200:
        return f"[Error: HTTP {resp.status_code} from xai {model}]"

    data = resp.json()
    if "error" in data:
        return f"[Error: {data['error'].get('message', 'Unknown')}]"

    choices = data.get("choices", [])
    if not choices:
        return f"[Error: No response from xai {model}]"

    content = (choices[0].get("message") or {}).get("content", "").strip()
    return _strip_think(content) if content else f"[No response from xai {model}]"


# --- Unified query with fallback ---


async def query_with_fallback(
    client: httpx.AsyncClient,
    keys: dict[str, str | None],
    entry: ModelEntry,
    messages: list[Message],
    max_tokens: int = 4096,
    timeout: float = 90,
    effort: ReasoningEffort | None = None,
) -> tuple[str, str, str]:
    """Query a model with provider-specific fallback, returns (name, model_used, response)."""
    model_name = entry.model.rsplit("/", 1)[-1]
    cap_timeout = min(timeout, 120 if is_thinking_model(entry.model) else 60)

    # Try native provider first if fallback is specified
    if entry.fallback:
        provider, fb_model = entry.fallback

        if provider == "anthropic" and keys.get("anthropic"):
            # Try claude --print first (uses Max subscription)
            response = await query_claude_print(fb_model, messages, cap_timeout)
            if not is_error(response):
                return (entry.name, fb_model, response)
            # Then try Anthropic API
            response = await query_anthropic(
                client, keys["anthropic"], fb_model, messages, max_tokens, cap_timeout, effort
            )
            if not is_error(response):
                return (entry.name, fb_model, response)

        elif provider == "zhipu" and keys.get("zhipu"):
            response = await query_bigmodel(
                client, keys["zhipu"], fb_model, messages, max_tokens, cap_timeout
            )
            if not is_error(response):
                return (entry.name, fb_model, response)

        elif provider == "xai" and keys.get("xai"):
            response = await query_xai(
                client, keys["xai"], fb_model, messages, max_tokens, cap_timeout, effort
            )
            if not is_error(response):
                return (entry.name, fb_model, response)

        elif provider == "google" and keys.get("google"):
            response = await query_google(
                client, keys["google"], fb_model, messages, max_tokens, cap_timeout, effort
            )
            if not is_error(response):
                return (entry.name, fb_model, response)

    # Fall back to OpenRouter
    if keys.get("openrouter"):
        response = await query_openrouter(
            client, keys["openrouter"], entry.model, messages, max_tokens, cap_timeout, effort
        )
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
    """Query all models in parallel, returns list of (name, model_used, response)."""
    keys = api_keys()
    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.wait_for(
                query_with_fallback(client, keys, entry, messages, max_tokens, timeout, effort),
                timeout=timeout,
            )
            for entry in models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            name = models[idx].name
            output.append((name, name, f"[Error: {name} timed out]"))
        else:
            output.append(result)
    return output


async def query_judge(
    model: str,
    messages: list[Message],
    max_tokens: int = 16384,
    timeout: float = 300,
    effort: ReasoningEffort | None = None,
) -> str:
    """Query the judge model with native-first fallback."""
    keys = api_keys()
    async with httpx.AsyncClient() as client:
        # Try native provider first
        if ("google/" in model or "gemini" in model) and keys.get("google"):
            bare = model.removeprefix("google/")
            response = await query_google(
                client, keys["google"], bare, messages, max_tokens, timeout, effort
            )
            if not is_error(response):
                return response

        if ("anthropic/" in model or "claude" in model):
            response = await query_claude_print(model, messages, timeout)
            if not is_error(response) :
                return response
            if keys.get("anthropic"):
                response = await query_anthropic(
                    client, keys["anthropic"], model, messages, max_tokens, timeout, effort
                )
                if not is_error(response):
                    return response

        # Fallback to OpenRouter
        if keys.get("openrouter"):
            return await query_openrouter(
                client, keys["openrouter"], model, messages, max_tokens, timeout, effort
            )

    return f"[Error: No API keys available for judge {model}]"
