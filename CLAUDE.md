# quorate — Project Instructions

Multi-model deliberation CLI in Python. Queries 6 frontier LLMs, runs structured debate, judge synthesises.

## Build & Test

```bash
uv venv && uv pip install -e . && uv pip install pytest
uv run pytest assays/ -x -v
```

## Architecture

- `src/quorate/config.py` — model definitions, API key resolution (pure env vars, injected by `op run`), constants
- `src/quorate/api.py` — HTTP clients (native provider first, OpenRouter fallback), parallel queries
- `src/quorate/prompts.py` — all prompt templates
- `src/quorate/modes/quick.py` — parallel queries, no debate
- `src/quorate/modes/council.py` — blind → debate → judge → critique
- `src/quorate/cli.py` — cyclopts subcommand CLI with preset system

## Key Patterns

- **Provider routing:** `_detect_provider()` → try native API (Google/Anthropic/xAI/OpenAI/ZhiPu) → OpenRouter fallback
- **Claude via `claude --print`** (Max subscription, $0) — first fallback for Anthropic models
- **GPT via `codex exec`** (Pro subscription, $0) — first fallback for OpenAI models, uses `model_reasoning_effort="xhigh"`
- **Gemini via `gemini -p`** (Gemini subscription, $0) — first fallback for Google models
- **ZhiPu native** — direct API at `open.bigmodel.cn/api/coding/paas/v4` (coding plan, free GLM-5.1)
- **API keys via `op run`:** effector wrapper injects keys from `quorate.env.op` at startup — no 1Password code in Python
- **Presets:** redteam/premortem/oxford/discuss are thin wrappers over council with preset context prompts

## Models (7)

| Model | Provider | Display Name |
|-------|----------|-------------|
| google/gemini-3.1-pro-preview | gemini -p / Google AI Studio | Gemini-3.1-Pro |
| openai/gpt-5.5 | Codex exec | GPT-5.5 |
| anthropic/claude-opus-4-7 | claude --print | Claude-Opus-4-7 |
| x-ai/grok-4.20-0309-reasoning | xAI native | Grok-4.20β |
| moonshotai/kimi-k2.6 | OpenRouter | Kimi-K2.6 |
| glm-5.1 | ZhiPu native | GLM-5.1 |
| xiaomi/mimo-v2.5-pro | OpenRouter | MiMo-V2.5-Pro |

## Runtime expectations

- `quorate quick` — 7 parallel queries, ~13-15s total
- `quorate council --fast` — blind + judge only, skips debate + critic, **~2-3 min** (judge with HIGH reasoning effort dominates)
- `quorate council` — full 4-phase deliberation, **5-8 min** (default), **12-15 min** with `--deep`

**Caller timeout guidance:**
- For `council`: outer timeout 600s+ (default), 900s+ with `--deep`
- From CC or other agents: prefer `run_in_background: true` and read the output file when the task completes
- Don't run full council on short inputs (<one page) — use `quorate quick` + manual synthesis, or `--fast`

## Gotchas

- OpenRouter returns 403 for OpenAI/Google/Anthropic models from HK IP — must use native APIs
- `codex exec` needs `-c 'model_reasoning_effort="xhigh"'` and `--skip-git-repo-check`
- `codex exec` strips `-pro` suffix from model name (uses base model names). `gpt-5.5-pro` is gated to ChatGPT Pro/Business/Enterprise tiers; ChatGPT Plus accounts get a 400 "not supported when using Codex with a ChatGPT account" (verified 2026-04-26). For Plus accounts: use `gpt-5.5`. For Pro variant access: route via OpenRouter (`openai/gpt-5.5-pro` triggers fallback) or upgrade Codex auth to Pro tier.
- Gemini CLI needs `GEMINI_HOME=~/.gemini-headless` to skip hooks (stdout pollution + latency)
- Gemini `-o json` output may have prepended text — scan for first `{`
- Thinking models need longer timeouts (180s+) in parallel execution
- Council runtime is dominated by sequential debate phase (~2-3 min for 7 speakers) + judge with 300s timeout — not a bug, just the design
