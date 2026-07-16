# quorate — Project Instructions

Multi-model deliberation CLI in Python. Queries 6 frontier LLMs, runs structured debate, judge synthesises.

## Build & Test

```bash
uv venv && uv pip install -e . && uv pip install pytest
uv run pytest assays/ -x -v
scripts/install-local.sh
```

## Architecture

- `src/quorate/config.py` — model definitions, API key resolution (pure env vars, injected by `op run`), constants
- `src/quorate/api.py` — HTTP clients (native provider first, OpenRouter fallback), parallel queries
- `src/quorate/prompts.py` — all prompt templates
- `src/quorate/modes/quick.py` — parallel queries, no debate
- `src/quorate/modes/council.py` — blind → debate → judge → critique
- `src/quorate/benchmark.py` — fixed synthetic roster canaries and dated health snapshots
- `src/quorate/cli.py` — cyclopts subcommand CLI with preset system

## Key Patterns

- **Provider routing:** `_detect_provider()` → try native API (Google/Anthropic/xAI/OpenAI/ZhiPu) → OpenRouter fallback
- **Claude via `claude --print`** (Max subscription, $0) — first fallback for Anthropic models
- **GPT via `codex exec`** (Pro subscription, $0) — first fallback for OpenAI models, uses `model_reasoning_effort="xhigh"`
- **Gemini via `gemini -p`** (Gemini subscription, $0) — first fallback for Google models
- **ZhiPu native** — direct API at `open.bigmodel.cn/api/coding/paas/v4` (coding plan, zero marginal-cost GLM-5.2)
- **API keys via `op run`:** effector wrapper injects keys from `quorate.env.op` at startup — no 1Password code in Python
- **Executable split:** `~/.local/bin/quorate` links to the effector; the effector calls `quorate-core` after injecting keys, preventing both credential bypass and wrapper recursion
- **Presets:** redteam/premortem/oxford/discuss are thin wrappers over council with preset context prompts

## Models (7)

| Model | Provider | Display Name |
|-------|----------|-------------|
| google/gemini-3.1-pro-preview | gemini -p / Google AI Studio | Gemini-3.1-Pro |
| openai/gpt-5.6-sol | Codex exec | GPT-5.6-Sol |
| anthropic/claude-fable-5 | claude --print | Claude-Fable-5 |
| x-ai/grok-4.5 | xAI native | Grok-4.5 |
| moonshotai/kimi-k2.6 | OpenRouter | Kimi-K2.6 |
| z-ai/glm-5.2 | ZhiPu native | GLM-5.2 |
| xiaomi/mimo-v2.5-pro | OpenRouter | MiMo-V2.5-Pro |

Judge: Gemini 3.1 Pro, with GPT-5.6 Sol through Codex as the fallback when all Gemini routes fail. Critic: Claude Opus 4.8.

## Runtime expectations

- `quorate quick` — 7 parallel queries; subscription routes get 120 seconds before fallback
- `quorate benchmark` — 3 sequential synthetic canaries across all 7 quick seats
- `quorate council --fast` — blind + judge only, skips debate + critic, **~2-3 min** (judge with HIGH reasoning effort dominates)
- `quorate council` — full 4-phase deliberation, **5-8 min** (default), **12-15 min** with `--deep`

**Caller timeout guidance:**
- For `council`: outer timeout 600s+ (default), 900s+ with `--deep`
- From CC or other agents: prefer `run_in_background: true` and read the output file when the task completes
- Don't run full council on short inputs (<one page) — use `quorate quick` + manual synthesis, or `--fast`

## Gotchas

- OpenRouter returns 403 for OpenAI/Google/Anthropic models from HK IP — must use native APIs
- `codex exec` needs `-c 'model_reasoning_effort="xhigh"'` and `--skip-git-repo-check`
- `codex exec` strips an OpenRouter-style `-pro` suffix. Use the actual Codex model ID, currently `gpt-5.6-sol`; quick mode requests medium reasoning while council retains the CLI's highest default when no effort is supplied.
- Gemini CLI strips API-key variables so the existing subscription login wins. Its `-o json` output may have prepended hook text, so scan for the first `{`.
- Thinking models need longer timeouts (180s+) in parallel execution
- Scripted quick and council runs fail closed when fewer than a strict majority respond. Their error envelope retains partial responses plus sanitized route codes, never provider error prose.
- Benchmark snapshots never persist response text and never edit the roster. A seat change requires agreement between external task-specific evidence and local canaries.
- Council runtime is dominated by sequential debate phase (~2-3 min for 7 speakers) + judge with 300s timeout — not a bug, just the design
