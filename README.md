# quorate

Multi-model deliberation CLI — 6 frontier LLMs debate, then judge.

Named after [quorum sensing](https://www.nature.com/articles/s41467-023-37950-7): bacteria pool imperfect estimates to make better collective decisions. quorate does the same with language models.

## Install

```bash
uvx quorate quick "Should I learn Rust?"
# or
pip install quorate
```

On the Vivesca host, install through the local helper so the ordinary `quorate`
command enters the 1Password effector and can use subscription and API routes:

```bash
scripts/install-local.sh
```

## Models

The six council debaters (`resolved_council()` in `config.py`):

| Model | Provider (native → fallback) |
|-------|------------------------------|
| GPT-5.6 Sol | OpenAI Codex CLI → OpenAI API → OpenRouter |
| Claude Fable 5 | Claude CLI → Anthropic API → OpenRouter |
| Grok 4.5 | xAI API → OpenRouter |
| Kimi K2.6 | OpenRouter (Moonshot) |
| GLM-5.2 | ZhiPu API → OpenRouter |
| MiMo v2.5 Pro | OpenRouter (Xiaomi) |

Judge: Gemini 3.5 Flash (Gemini CLI → Google AI Studio → OpenRouter), with GPT-5.6 Sol through the Codex subscription as the cross-vendor fallback. Critic: Claude Opus 4.8.

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

# Synthetic roster health check; saves a dated local snapshot
quorate benchmark

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

## Data boundary

Quorate sends the question, context, and persona to multiple external model providers. Treat
every input as an external disclosure. Do not submit credentials, personal data, confidential
client material, private communications, or proprietary source text. Removing names alone is
not sufficient sanitization. Persona files should be public, synthetic, or deliberately
sanitized.

Set `QUORATE_PROTECTED_ROOTS` to an OS-path-separator-delimited list of local directories that
must never be read as Quorate inputs:

```bash
export QUORATE_PROTECTED_ROOTS="$HOME/private-client:$HOME/private-personal"
```

Question, context, and persona files beneath those roots are rejected, including symlinks that
resolve into them. This is a guardrail for known directories; it cannot classify inline text or
prove that a copied file has been safely sanitized.

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

GPT-5.6 Sol uses [Codex CLI](https://github.com/openai/codex) (`codex exec`), Claude uses `claude --print`, and Gemini uses the Gemini CLI (`gemini -p`) — all route through their respective subscriptions at zero marginal cost, falling back to the direct API and then OpenRouter. Telemetry records the model and route actually used, and subscription-backed calls are not priced as API usage.

Scripted runs require a strict majority of configured seats. Quick mode therefore needs four of seven successful responses, while council needs four of six in its blind phase. A degraded run returns a non-zero JSON error envelope containing the partial responses and safe route diagnostics such as `http_404`, `timeout`, or `no_credentials`; provider prose and secrets are never copied into diagnostics.

## Roster review policy

`quorate benchmark` runs three fixed synthetic canaries for route availability,
strict structured output, and simple deterministic reasoning. It stores no response
text. Dated snapshots under `~/.local/state/quorate/benchmarks/` contain only the
route used, latency, pass state, and safe diagnostics.

Treat Artificial Analysis as the broad screening layer for intelligence, cost,
speed, and provider performance. Cross-check material changes against a task-specific
source such as Epoch AI, Arena, or SWE-bench, then require agreement with the local
canaries before changing a seat. One failed seat is tolerated when every canary still
has a strict-majority quorum. Quorate never edits its own roster from benchmark output.

On the Vivesca host, `scripts/monthly-benchmark.sh` is the non-interactive runner.
It loads the locally resolved credential environment, uses subscription routes first,
and stays silent when the roster is healthy. The corresponding LaunchAgent runs at
09:00 on the first day of each month and writes only degraded or failed output to
`~/logs/quorate-monthly-benchmark.log`.

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
