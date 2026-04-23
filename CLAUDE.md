# quorate — Project Instructions

Multi-model deliberation CLI in Python. Queries 6 frontier LLMs, runs structured debate, judge synthesises.

## Build & Test

```bash
uv venv && uv pip install -e . && uv pip install pytest
uv run pytest assays/ -x -v
```

## Architecture

- `src/quorate/config.py` — model definitions, API key resolution (env → 1Password fallback), constants
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
- **ZhiPu native** — direct API at `open.bigmodel.cn/api/paas/v4`, free tier for GLM models
- **1Password auto-resolve:** `_op_read()` in config.py fills missing API keys from Agents vault
- **Presets:** redteam/premortem/oxford/discuss are thin wrappers over council with preset context prompts

## Models (6)

| Model | Provider | Display Name |
|-------|----------|-------------|
| google/gemini-3.1-pro-preview | gemini -p / Google AI Studio | Gemini-3.1-Pro |
| openai/gpt-5.4-pro | Codex exec | GPT-5.4-Pro |
| anthropic/claude-opus-4-6 | claude --print | Claude-Opus-4-6 |
| x-ai/grok-4.20-0309-reasoning | xAI native | Grok-4.20β |
| qwen/qwen3.6-plus | OpenRouter | Qwen3.6-Plus |
| z-ai/glm-5.1 | ZhiPu native | GLM-5.1 |

## Gotchas

- OpenRouter returns 403 for OpenAI/Google/Anthropic models from HK IP — must use native APIs
- `codex exec` needs `-c 'model_reasoning_effort="xhigh"'` and `--skip-git-repo-check`
- `codex exec` strips `-pro` suffix from model name (uses base model names)
- `op item get` works with parentheses in names; `op read` doesn't
- Thinking models need longer timeouts (180s+) in parallel execution
