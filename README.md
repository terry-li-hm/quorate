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

The six council debaters (`resolved_council()` in `config.py`):

| Model | Provider (native → fallback) |
|-------|------------------------------|
| GPT-5.5 | OpenAI Codex CLI → OpenAI API → OpenRouter |
| Claude Opus 4.7 | Claude CLI → Anthropic API → OpenRouter |
| Grok 4.3 | xAI API → OpenRouter |
| Kimi K2.6 | OpenRouter (Moonshot) |
| GLM-5.1 | ZhiPu API → OpenRouter |
| MiMo v2.5 Pro | OpenRouter (Xiaomi) |

Judge: Gemini 3.1 Pro (Gemini CLI → Google AI Studio → OpenRouter). Critic: Claude Opus 4.7.

Any seat is overridable via `CONSILIUM_MODEL_M1`…`M6`, `CONSILIUM_MODEL_JUDGE`, and `CONSILIUM_MODEL_CRITIQUE`.

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

Council takes five flags — everything else is injected through the question or `--context`:

```bash
quorate council "question"                       # default: 1 debate round + critique (5-8 min)
quorate council "question" --fast                # skip debate + critique (~2-3 min)
quorate council "question" --deep                # 2 debate rounds (12-15 min)
quorate council "question" --persona ./founder.md  # all models answer as this principal (file path)
quorate council "question" --context banking-frame.md  # inject regulatory/domain context
quorate council "question" --json                # force JSON envelope (auto when piped)
```

`--persona` takes a **file path** to a stakeholder profile, not an inline description — every
model then debates in that principal's first-person voice. `--context` is repeatable; each item
is read as a file if it exists, else treated as inline text. There is no `--rounds`, `--no-critic`,
or `--domain` flag: use `--fast`/`--deep` for round count, and put regulatory framing in the
question or a `--context` file. The presets (`redteam`, `premortem`, `oxford`, `discuss`) accept
only `--context` and `--json`; `--persona` is `council`/`quick` only.

## API Keys

quorate routes through native provider APIs first, falling back to OpenRouter:

```bash
export GOOGLE_API_KEY="..."           # Gemini (Google AI Studio)
export XAI_API_KEY="..."              # Grok (xAI)
export ZHIPU_API_KEY="..."            # GLM (ZhiPu native)
export ANTHROPIC_API_KEY="..."        # Claude (fallback if no claude CLI)
export OPENAI_API_KEY="..."           # GPT (fallback if no Codex CLI)
export OPENROUTER_API_KEY="..."       # Kimi, MiMo, and fallback for all
export QUORATE_OPENROUTER_KEY="..."   # Dedicated OpenRouter key (takes priority)
```

GPT-5.5 uses [Codex CLI](https://github.com/openai/codex) (`codex exec`), Claude uses `claude --print`, and Gemini uses the Gemini CLI (`gemini -p`) — all route through their respective subscriptions at zero per-token cost, falling back to the direct API and then OpenRouter.

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
