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
| Claude Opus 4.8 | Claude CLI → Anthropic API → OpenRouter |
| Grok 4.5 | xAI API → OpenRouter |
| Kimi K3 | Kimi Code CLI subscription |
| GLM-5.2 | ZhiPu API → OpenRouter |
| DeepSeek V4 Pro | OpenRouter (DeepSeek) |

Judge: Claude Fable 5 (Claude CLI → Anthropic API → OpenRouter), with GPT-5.6 Sol through the Codex subscription as the cross-vendor fallback. Critic: Gemini 3.5 Flash (Antigravity CLI → Google AI Studio → OpenRouter), with Claude Opus 4.8 as its fallback.

Brainstorm mode uses the six council families plus Gemini 3.5 Flash and MiniMax M3. Claude Fable curates the resulting ideas but does not generate them.

Any seat is overridable via `CONSILIUM_MODEL_M1`…`M6`, `CONSILIUM_MODEL_JUDGE`, `CONSILIUM_MODEL_JUDGE_FALLBACK`, `CONSILIUM_MODEL_CRITIQUE`, and `CONSILIUM_MODEL_CRITIQUE_FALLBACK`.

## Usage

```bash
# Parallel queries — all models answer independently
quorate quick "What makes a good CLI?"

# Full deliberation — blind phase, debate, judge, critique
quorate council "Should we rewrite our backend in Rust?"

# Divergent ideation — eight lenses, cross-pollination, curated shortlist
quorate brainstorm "Ideas for a calmer personal knowledge archive"

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
only `--context` and `--json`; `brainstorm` also accepts those two flags, while `--persona`
is `council`/`quick` only.

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
export OPENROUTER_API_KEY="..."       # DeepSeek and API fallback routes
export QUORATE_OPENROUTER_KEY="..."   # Dedicated OpenRouter key (takes priority)
```

GPT-5.6 Sol uses [Codex CLI](https://github.com/openai/codex) (`codex exec`), Claude uses `claude --print`, Gemini uses Antigravity (`agy --print`), and K3 uses Kimi Code prompt mode. These routes use their respective subscriptions at zero marginal cost. Each subscription CLI receives an allowlisted environment without unrelated provider credentials and runs from a temporary workspace; Claude has no tools, Codex is read-only, and Antigravity uses sandboxed plan mode. GPT, Claude, and Gemini can fall back to a direct API and then OpenRouter; K3 fails closed when the subscribed CLI route is unavailable. Telemetry records the model and route actually used, and subscription-backed calls are not priced as API usage.

The [subscription CLI isolation audit](docs/experiments/2026-07-17-subscription-cli-isolation.md)
records the ambient-authority finding, containment changes, and live route verification.

Scripted runs require a strict majority of configured seats. Quick mode therefore needs four of seven successful responses, council needs four of six in its blind phase, and brainstorm needs five of eight independent generators. A degraded run returns a non-zero JSON error envelope containing the partial responses and safe route diagnostics such as `http_404`, `timeout`, or `no_credentials`; provider prose and secrets are never copied into diagnostics.

## Roster review policy

`quorate benchmark` runs three fixed synthetic canaries across every primary production role,
including the critic, for route availability,
strict structured output, and simple deterministic reasoning. It stores no response
text. Dated snapshots under `~/.local/state/quorate/benchmarks/` contain only the
route used, latency, pass state, and safe diagnostics.

Treat Artificial Analysis as the broad screening layer for intelligence, cost,
speed, and provider performance. Cross-check material changes against a task-specific
source such as Epoch AI, Arena, or SWE-bench, then require agreement with the local
canaries before changing a seat. One failed seat is tolerated when every canary still
has a strict-majority quorum. Quorate never edits its own roster from benchmark output.

`quorate usage --days 30` summarizes the response-free run log by model, route,
reachability, mean and p95 latency, and estimated API cost. It writes a dated snapshot under
`~/.local/state/quorate/usage/`. The monthly roster job records this snapshot without invoking
another model, so membership value can be reviewed from actual use rather than synthetic assays.

Material role changes also receive a durable experiment note. See the
[2026-07-16 judge role selection](docs/experiments/2026-07-16-judge-role-selection.md)
for the evidence behind the Fable judge architecture, the
[2026-07-17 Fable versus GPT judge assay](docs/experiments/2026-07-17-fable-vs-gpt-judge-assay.md)
for the direct retention test, the
[2026-07-17 K3 versus K2.6 council-seat evaluation](docs/experiments/2026-07-17-k3-vs-k2.6-council-seat.md)
for the Kimi-seat replacement, the
[2026-07-17 K3 downstream synthesis replay](docs/experiments/2026-07-17-k3-downstream-replay.md)
for the deliberately inconclusive downstream follow-up, and the
[2026-07-16 DeepSeek seat selection](docs/experiments/2026-07-16-deepseek-seat-selection.md)
for the evidence behind the sixth council seat, and the
[2026-07-16 brainstorm mode validation](docs/experiments/2026-07-16-brainstorm-mode.md)
for the divergent ideation architecture.

On the Vivesca host, `scripts/monthly-benchmark.sh` is the non-interactive runner.
It loads the locally resolved credential environment, uses subscription routes first,
records the rolling usage snapshot, and stays silent when the roster is healthy. The corresponding LaunchAgent runs at
09:00 on the first day of each month and writes only degraded or failed output to
`~/logs/quorate-monthly-benchmark.log`.

## How it works

**Quick mode**: Fan out the question to all models in parallel. Collect and display.

**Brainstorm mode**: Eight model families generate through distinct lenses without seeing one another, cross-pollinate once with a neighboring model, then Claude Fable clusters duplicates and returns six ideas plus one wildcard.

**Council mode**:
1. **Blind phase** — all models stake positions independently (prevents anchoring)
2. **Debate** — sequential rounds with a rotating challenger who must dissent
3. **Judge** — Claude Fable synthesizes competing arguments into a recommendation
4. **Critique** — Gemini reviews the judge's synthesis for blind spots

**Presets** (redteam, premortem, oxford, discuss) are council with preset system prompts and flags. No separate code paths.

## License

MIT
