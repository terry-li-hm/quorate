# quorate

Multi-model deliberation CLI — 6 frontier LLMs debate, then judge.

Named after [quorum sensing](https://www.nature.com/articles/s41467-023-37950-7): bacteria pool imperfect estimates to make better collective decisions. quorate does the same with language models.

## Install

```bash
uvx quorate quick "Should I learn Rust?"
# or
pip install quorate
```

## Models

| Model | Provider |
|-------|----------|
| Gemini 3.1 Pro | Google AI Studio |
| GPT-5.4 | OpenAI (Codex CLI) |
| Claude Opus 4.6 | Anthropic (Claude CLI) |
| Grok 4.20β | xAI |
| DeepSeek V3.2 | OpenRouter |
| GLM-5.1 | OpenRouter |

Judge: Gemini 3.1 Pro. Critic: Claude Sonnet 4.6.

## Usage

```bash
# Parallel queries — all models answer independently
quorate quick "What makes a good CLI?"

# Full deliberation — blind phase, debate, judge, critique
quorate council "Should we rewrite our backend in Rust?"

# Adversarial stress-test
quorate redteam "Our launch plan for Q3"

# Assume failure, write past-tense narratives
quorate premortem "Migration to microservices"

# Binary FOR vs AGAINST debate
quorate oxford "AI will replace most knowledge work within 5 years"

# Open roundtable, no judge
quorate discuss "The future of open source"

# Auto-classify (no subcommand)
quorate "What's the best database for time-series data?"

# Read question from file
quorate council /path/to/prompt.txt
```

### Council options

```bash
quorate council "question" --deep          # 2 debate rounds
quorate council "question" --rounds 3      # custom rounds
quorate council "question" --no-critic     # skip critique phase
quorate council "question" --domain banking # regulatory context
quorate council "question" --persona "startup founder with $2M runway"
```

## API Keys

quorate routes through native provider APIs first, falling back to OpenRouter:

```bash
export GOOGLE_API_KEY="..."           # Gemini (Google AI Studio)
export XAI_API_KEY="..."              # Grok (xAI)
export ANTHROPIC_API_KEY="..."        # Claude (fallback if no claude CLI)
export OPENROUTER_API_KEY="..."       # DeepSeek, GLM, fallback for all
export QUORATE_OPENROUTER_KEY="..."   # Dedicated OpenRouter key (takes priority)
```

GPT-5.4 uses [Codex CLI](https://github.com/openai/codex) (`codex exec`). Claude uses `claude --print`. Both route through their respective subscriptions at zero per-token cost.

## How it works

**Quick mode**: Fan out the question to all models in parallel. Collect and display.

**Council mode**:
1. **Blind phase** — all models stake positions independently (prevents anchoring)
2. **Debate** — sequential rounds with a rotating challenger who must dissent
3. **Judge** — Gemini synthesizes competing arguments into a recommendation
4. **Critique** — Claude reviews the judge's synthesis for blind spots

**Presets** (redteam, premortem, oxford, discuss) are council with preset system prompts and flags. No separate code paths.

## License

MIT
