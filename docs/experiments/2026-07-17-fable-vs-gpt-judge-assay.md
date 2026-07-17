# Fable versus GPT judge assay

Date: 2026-07-17

Status: retain Claude Fable 5 as judge

## Question

Should GPT-5.6 Sol replace Claude Fable 5 as Quorate's final judge?

Fable is the incumbent judge and GPT is its cross-vendor fallback. GPT also holds a council seat, so replacing Fable would reduce independence between the deliberation and the final synthesis. The predeclared promotion rule therefore required GPT to lead by at least two usable correct cases while completing at least 19 of 20 output contracts. A tie or one-case lead would retain Fable.

## Method

Twenty balanced, labeled synthetic cases tested deterministic judgment across capacity, privacy, expected value, reliability, contracts, statistics, operational risk, access control, scheduling, and measurement. Ten cases expected recommendation A and ten expected B. Both models received identical evidence, a 1,024-token limit, high reasoning effort, and a strict final-decision marker. Four cases ran concurrently.

A comparison was valid only when both routes returned at least 18 reachable, contract-compliant results. This prevents a provider outage from being misread as a capability difference. Inputs contained no private data, and the report retained no response text.

## Route repair

The first attempt was invalid because Fable returned no reachable responses while GPT completed all 20. A minimal reproduction identified an expired Claude OAuth session. Reauthorization then exposed a local test fixture at `~/.claude/.credentials.json` containing `{"key":"val"}`, which masked the new credential. The fixture was preserved under an invalid-fixture backup name, Claude's stale credential was cleared through its own logout command, and a clean login restored the route. An exact `ROUTE_OK` probe passed before the assay was rerun. The invalid run was excluded from the capability comparison.

## Result

Fable was reachable in 20 of 20 cases, completed all 20 contracts, and returned all 20 correct decisions. Its mean latency was 14.73 seconds and its median was 14.55 seconds through `claude-print`.

GPT was also reachable in 20 of 20 cases, completed all 20 contracts, and returned all 20 correct decisions. Its mean latency was 14.62 seconds and its median was 13.54 seconds through `codex-exec`.

The models produced the same correctness outcome on every case. GPT's 0.11-second mean latency advantage is operationally immaterial and was secondary to judgment quality.

## Decision

Retain Claude Fable 5 as judge and GPT-5.6 Sol as fallback. GPT did not clear the predeclared two-case quality margin, while Fable preserves independence from GPT's council contribution. No production routing change is warranted.

The aggregate report is stored outside the repository at `~/.local/state/quorate/judge-assays/2026-07-17-fable-vs-gpt-valid.json`. The reusable assay and its validity gate remain in the repository for future role reviews.

## Limitations

The suite contains only 20 synthetic cases and emphasizes deterministic decisions with known labels. It does not measure synthesis quality on natural council transcripts, resistance to persuasion by a model's own contribution, or performance on ambiguous recommendations. The result supports retaining the current role assignment; it does not establish general equivalence between the models.
