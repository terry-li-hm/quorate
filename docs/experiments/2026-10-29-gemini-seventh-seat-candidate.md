# Gemini seventh council seat: candidate, route gate, and predeclared assay

Date: 2026-10-29

## Status

Carbon was assayed on 16 November 2026 and failed only the latency floor, so the roster stays unchanged. See
[Carbon result](#carbon-result-16-november-2026) below. Terry approved running Argon, the fallback candidate,
at about 11:00 HKT, and its run started at 11:03 HKT under the same harness and rule. Terry
asked "run the gemini seat test?" that morning, after the K3 seat was found dead. The harness and the admission
rule below are unchanged.

What changed since 29 October, all checked on Ganglion on 16 November:

- **Route.** `agy models` lists Gemini 4 Carbon and Argon at Low, Medium and High, and both answered a
  synthetic prompt through `agy --print` in plan mode, Carbon in 37.7 s and Argon in 27.1 s. Commit
  `47aaf36` maps both to those exact labels at Medium by default, as the Carbon plan below allows.
- **Candidate.** Carbon, under the Carbon plan below, because it now answers on a route Quorate can use
  and is Vivesca's default cross-family challenger. Argon stays the fallback candidate if Carbon fails
  only the latency floor.
- **Judges.** Claude Fable 5.2 and GPT-6.1 Sol both answered the 16 November speedtest.
- **Incumbents.** Seats are GPT-6.1 Sol, Claude Opus 5.5, Grok 4.7, K3, GLM-5.3 and DeepSeek V4 Pro.
  K3 has refused every call since at least 1 October, after the Kimi membership was cancelled on
  20 August, so it scores nothing and the run stays valid only if all five other seats answer at least
  18 cases. The incumbent median is then taken over five seats.
- **Critic route.** Antigravity no longer accepts the label `Gemini 3.5 Flash (Medium)`, so the critic
  falls back to Google AI Studio. Separate from this assay and not yet decided. On 16 November `agy models`
  listed Gemini 3.8 Flash and Gemini 4 Hydrogen at several levels and no 3.5 Flash. Press reports describe
  Hydrogen as an unannounced Flash-tier Gemini 4, so both are critic candidates, and a switch is a role change
  under the [model role governance policy](../model-role-governance.md).

The run started at 10:31 HKT as a detached process with `--candidate google/gemini-4-carbon` and finished at
10:51 HKT. Its report is `~/.local/state/quorate/council-seat-assays/2026-11-16-gemini-4-carbon-seat.json`,
with its log beside it under a `.log` suffix.

## Question

Should the council gain a seventh, Google-family seat? The six seats cover OpenAI, Anthropic, xAI, Moonshot,
Zhipu, and DeepSeek. Gemini serves only as the critic and as one of eight brainstorm generators, so the
convergent council has no Google voice.

## External evidence

Google announced Gemini 4 Carbon on 28 October 2026 as its strongest model. It is available only to a subset
of vetted cyber defenders through Google's Fairwind program, with no API, model ID, or wider-release date.
On 29 October the Gemini API returned `NOT_FOUND` for `gemini-4-carbon`, and OpenRouter did not list it.
Source: [Gemini 4 Carbon announcement](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-carbon/), retrieved 2026-10-29.

Carbon builds on Gemini 4 Argon, which Google released to developers on 13 October 2026. The Gemini API
lists `gemini-4-argon` with a 1,048,576-token input limit, a 32,768-token output limit, and thinking.
OpenRouter lists it at USD 2 input and USD 10 output per million tokens. Launch coverage reports that this
introductory price ends on 30 November, when list pricing becomes USD 4 and USD 20. Google says Carbon will
launch at Argon's introductory price.

Broad ranking is screening evidence only. VentureBeat reported Artificial Analysis Intelligence Index v5.0
scores of 59 for Carbon, 58 for Claude Opus 5.5, and 57 for GPT-6 Astra. Source:
[VentureBeat](https://venturebeat.com/technology/google-unveils-gemini-4-carbon-retaking-the-worlds-no-1-spot-on-updated-harder-benchmarks-but-access-is-limited), retrieved 2026-10-29.

## Route probes

Each probe on 29 October used a one-line synthetic prompt and retained no response text.

- **Google AI Studio.** Quorate's key is on the free tier. Metadata for `gemini-4-argon` returned 200, but
  generation returned 429 with free-tier request and token quota violations: Argon has no free-tier
  allowance. The same key served `gemini-3.5-flash`, which reported `modelVersion` `gemini-3.6-flash`.
- **OpenRouter.** `google/gemini-4-argon` returned 403, "prohibited due to a violation of provider Terms Of
  Service", from the Tokyo host. This matches the existing block on Google, OpenAI, and Anthropic models for
  this account.
- **Antigravity.** `agy models` on Astrocyte reported that the CLI is signed out. Astrocyte's run log shows
  the last successful `antigravity-cli` Gemini call on 24 August. Since then, 13 Gemini 3.5 Flash calls fell
  back to Google AI Studio, the last on 1 October.

Every route to Argon therefore needs an account action by Terry: billing on the Gemini API project, or an
Antigravity sign-in on a plan that includes Argon.

The judges are also down. The 29 October canary on Ganglion, `~/.local/state/quorate/benchmarks/2026-10-29.json`,
was degraded with three failing roles, none of them touched by this change:

- **Claude Fable 5.1, the judge.** `claude --print` returns "Fable 5.1 requires usage credits", and the
  Anthropic API fallback returns HTTP 400. The 28 October council on Ganglion was already judged by the GPT
  fallback, and Astrocyte's 1 October canary shows the judge failing there too.
- **GPT-5.6 Sol, seat one and the judge fallback.** Codex on the ChatGPT account now rejects `gpt-5.6-sol`,
  `gpt-6-sol`, and `gpt-6.1-sol` as unsupported, although `gpt-5.6-sol` answered a council on 28 October.
  Codex's default `gpt-6-luna` still answers. The OpenAI API fallback returns HTTP 429.
- **K3, seat four.** The Kimi Code API returns HTTP 403, as it did on 28 October and on 1 October.

Until the judge or its fallback is restored or replaced, councils on Ganglion have no working judge, and the
assay preflight will stop any run as invalid.

## Harness defects found and fixed

Gemini's thinking tokens count against `maxOutputTokens`, and Quorate's Google client set that cap to the
visible-answer allowance alone. Measured on `gemini-3.5-flash` with a synthetic reasoning prompt:

| `maxOutputTokens` | `thinkingBudget` | Finish | Thought tokens | Answer tokens | Final marker |
|---|---|---|---|---|---|
| 1,024 | 4,096 | `MAX_TOKENS` | 973 | 128 | missing |
| 5,120 | 4,096 | `STOP` | 1,226 | 217 | present |
| 1,024 | unset | `MAX_TOKENS` | 852 | 168 | missing |

The council blind phase gives each seat 1,024 tokens, so Gemini answers were being cut off. The July
[judge role selection](2026-07-16-judge-role-selection.md) recorded eight truncated Gemini responses under a
1,024-token limit; this defect probably explains much of that result. It does not reopen the judge decision,
which stands until a material harness revision is assayed.

Commit `5d882c2` fixes the client and the routing around it:

- The Google client always sends a thinking budget and adds it to `maxOutputTokens`, so the visible answer
  keeps the caller's full allowance. It joins every non-thought text part, sends the key in a header instead
  of the URL, and reports a `MAX_TOKENS` finish as the `google-ai-studio:max_tokens` diagnostic.
- Antigravity maps only exact, verified model IDs. Any other Google model fails that route with the
  `antigravity-cli:unmapped_model` diagnostic and falls through to Google AI Studio with its exact ID.
  Previously every unrecognised model silently ran as Gemini 3.5 Flash while the run log recorded the
  requested model.
- `CONSILIUM_MODEL_M7` adds an optional seventh seat. It has no default, so the council stays at six seats
  until a trial sets it. Quorum remains a strict majority: four of seven.

After the fix was installed on Ganglion, the critic, Gemini 3.5 Flash, passed all three canaries through
Google AI Studio. A live run of the assay below then stopped at its preflight: Argon skipped Antigravity as
`unmapped_model` and failed on AI Studio with HTTP 429, rather than running as Flash.

## Predeclared assay

`src/quorate/seventh_seat_assay.py`, added in commit `4de3841`, asks whether the candidate contributes as
much distinct value to the panel as a typical existing seat. It reuses the 20 synthetic cases from the
[K3 council-seat evaluation](2026-07-17-k3-vs-k2.6-council-seat.md).

- **Seats.** The candidate and the six default seats each answer every case through Quorate's production
  routes with the council's blind-phase prompt, 1,024 visible tokens, and default effort. A configured
  seventh seat never enters the incumbent baseline.
- **Judges.** The council's configured judge and judge fallback at run time, which today are Claude Fable 5.1
  and GPT-5.6 Sol. A `--judge` pair can replace them, and the report names the judges used. Per case, each
  judge sees the candidate and every reachable incumbent, anonymised in a seeded order, with the hidden value
  criterion. They score each response's distinct contribution: 2 for an important, correct consideration no
  other response makes; 1 for a minor or partly distinct one; 0 for nothing distinct, or distinct content
  that is wrong. A case is judged only when the candidate and at least five incumbents answered.
- **Preflight.** One synthetic call each to the candidate and both judges runs first. If any fails, the run
  stops before any seat query and reports `invalid-run`, so a closed route can never produce a verdict.
- **Primary metric.** The candidate's mean score must be at least the median of the six incumbents' mean
  scores.
- **Floors, all non-compensable.** The candidate must be reachable in at least 19 of 20 cases, stay within
  60 to 220 words in at least 19, avoid `max_tokens` truncation in at least 19, and keep its median latency
  at or below the slowest incumbent's median, so it never becomes the council's bottleneck. A failed floor
  retains six seats.
- **Validity.** The run is invalid if fewer than five incumbents answer at least 18 cases, or, once the floors
  pass, if either judge completes fewer than 18 parseable judgments.
- **Retention.** The report keeps aggregates only and stores no model or judge text.

A pass admits the candidate to a trial, not to production. Run the module from the repository with the
resolved credential environment loaded, and write the report to
`~/.local/state/quorate/council-seat-assays/<date>-gemini-seventh-seat.json`.

## Trial and Carbon plan

After a pass, set `CONSILIUM_MODEL_M7=google/gemini-4-argon` as the reversible trial override and run
`quorate benchmark`. The seat must pass all three canaries with quorum. Natural use then decides
production under the [model role governance policy](../model-role-governance.md).

When Carbon answers on a route Quorate can use, set the seventh seat to Carbon's published ID, add an exact
Antigravity label only if `agy models` lists one, and rerun the canaries and this assay unchanged.

## Carbon result, 16 November 2026

The run returned `retain-six-seats`. Carbon cleared three of the four non-compensable floors and failed latency.
Under the locked rule a failed floor retains the roster whatever the score, so Carbon is not admitted to a trial.

The judges were Claude Fable 5.2 and GPT-6.1 Sol, and all three preflight calls answered without diagnostics.
The run is valid. Five incumbents answered at least 19 cases each, and K3 answered none, refused with HTTP 403
every time. Eighteen cases had the candidate and at least five incumbents and were judged. In the other two,
Carbon or GPT-6.1 Sol failed once. Each judge completed all 18 judgments, exactly the floor of 18.

| Seat | Reachable | 60–220 words | Truncated | Median latency (s) | Mean score |
|---|---|---|---|---|---|
| Gemini 4 Carbon, candidate | 19 | 19 | 0 | 75.4 | 1.194 |
| Claude Opus 5.5 | 20 | 20 | 0 | 6.7 | 1.306 |
| GPT-6.1 Sol | 19 | 19 | 0 | 43.4 | 1.222 |
| GLM-5.3 | 20 | 20 | 0 | 10.0 | 0.639 |
| Grok 4.7 | 20 | 20 | 0 | 17.3 | 0.444 |
| DeepSeek V4 Pro | 20 | 20 | 0 | 10.3 | 0.306 |
| K3 | 0 | 0 | 0 | none | none |

Carbon's mean of 1.194 is nearly twice the five-seat incumbent median of 0.639 and ranks third, behind Opus and
GPT. It would have passed the primary metric. Latency decided the verdict. Carbon's median was 75.4 s against the
slowest incumbent median, GPT-6.1 Sol's 43.4 s. Its fastest answer took 54.7 s, so none of its 19 answers came in
under that median. Its one failure hit Antigravity's 120-second subscription limit, then fell through to AI
Studio's free-tier 429 and OpenRouter's 403. GPT-6.1 Sol's one failure followed the same path through Codex.

While the run was going, the cockpit ran a 43-second single-process test suite and a few remote reads on the same
host. That cannot explain a gap that every Carbon answer exceeded, so the result stands.

Argon is the fallback candidate. Its one synthetic probe answered in 27.1 s
against Carbon's 37.7 s. Scaling Carbon's median by that ratio gives about 54 s, still above 43.4 s, so Argon may
fail the same floor. One probe is weak evidence, and only the run settles it. The roster and seat defaults are
unchanged.

Two consequences were corrected for Terry after the result. A slow seat costs more than its blind-phase gap,
because debate turns run one seat at a time, so a 75-second seat adds roughly its whole answer time to each round.
Carbon at its Low level is not a simple rerun: the assay queries seats at default effort, and a council's effort
applies to every seat, so it needs a per-seat effort setting and a newly predeclared assay.

## Argon run, 16 November 2026

Running. The process `python -I -m quorate.seventh_seat_assay --candidate google/gemini-4-argon` started at
11:03 HKT with the resolved credentials, the same way as Carbon's run, and passed its preflight within half a
minute. Its report goes to `~/.local/state/quorate/council-seat-assays/2026-11-16-gemini-4-argon-seat.json`,
with its log beside it under a `.log` suffix. The host's load average was about 6 on four cores at launch,
against about 3.5 during the Carbon run, from other sessions' work.

## Limitations

The default judges, Fable and GPT-5.6 Sol, score panels that include GPT-5.6 Sol and Claude Opus 5. Any
self-preference raises the incumbent median, which biases the assay against admission. Distinctiveness is measured against the day's six
incumbents, so a roster change elsewhere can move the bar. The client sends `thinkingBudget`, which works on
Gemini 3.5 Flash; whether Gemini 4 accepts it or requires `thinkingLevel` is unverified until the first
canary.
