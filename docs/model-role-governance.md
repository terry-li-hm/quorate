# Model role governance

Model admission is scoped to one role, route, and harness. A model that passes a Quorate council-seat evaluation is not thereby approved for coding, autonomous execution, mTOR, or any other system. Each role needs its own evidence and removal rule.

## Dormant since 16 November 2026

On 16 November 2026 Terry mothballed Quorate. Real councils became rare after August 2026, no run was ever rated for decision value, and upkeep kept landing: the judge broke in October, K3 refused every call for over six weeks before anyone noticed, the critic's Antigravity model label stopped working, and the latest seat assays took over an hour. Stopping maintenance loses nothing that cannot be restored, so the code, installs, run logs, benchmark snapshots and seat line-up stay in place, with Gemini 4 Argon still on trial in seat four.

Two things changed. The monthly seat benchmark, the LaunchAgent `com.vivesca.quorate-monthly-benchmark`, was booted out and disabled with its plist kept. Vivesca's skills stopped routing agents to Quorate, and the `quorate` skill became explicit-only, so it runs only when Terry names it.

To revive Quorate:

1. Re-enable and reload the monthly job with `launchctl enable gui/$(id -u)/com.vivesca.quorate-monthly-benchmark` and then `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.vivesca.quorate-monthly-benchmark.plist`.
2. Restore the skill routes by reverting or reapplying the Germline commit "receptors: stop offering Quorate; mark it dormant and explicit-only", which returns the `quorate` skill's routing description and triggers and the Quorate routes in the twelve skills that offered it.
3. Sign Antigravity back in on any host where it has lapsed, because seat four and the critic depend on it, then run `quorate benchmark` and treat any failed canary as a seat review under the rules below.

## Admission lifecycle

Every production role moves through four stages:

1. **Candidate.** External evidence shows that the model may improve the named role. Broad benchmark rank is a screening signal, not admission evidence.
2. **Assay.** A task-specific comparison changes one model or route at a time, uses synthetic and non-sensitive inputs, locks its primary metric and non-compensable validity floors before the run, and retains aggregate evidence rather than response text.
3. **Trial.** The candidate clears its assay and fixed route canaries, then enters the named role with a reversible override and response-free telemetry.
4. **Production.** Natural use confirms acceptable reachability and latency, while explicit outcome reviews support the role-value claim. An admission remains local to that role until another role independently passes the same lifecycle.

A role change requires external task-specific evidence, a valid local assay, healthy production canaries, and a durable experiment note. An inconclusive assay cannot establish an advantage. It may retain an incumbent when earlier evidence still holds, but it cannot broaden the model into another role.

## Review and removal

Review a role when its model or harness changes materially, monthly canaries show repeated degradation, natural usage contradicts the admission result, provider terms alter the permitted data boundary, or a better candidate has credible task-specific evidence. Suspend the route immediately if isolation fails, protected material may have been disclosed, or its failures prevent a strict-majority quorum. Retire it when a valid replacement assay wins or when sustained production value no longer justifies its cost and operational surface.

Do not rerun synthetic comparisons merely because a public benchmark looks stronger. Reopen a settled decision only after a material model or harness revision, contradictory natural evidence, or enough real cases to test a specific unresolved hypothesis.

## Data boundary

Quorate treats every question, context item, and persona as an external disclosure. Credentials, personal data, confidential client material, private communications, proprietary source text, and reconstructable internal context are prohibited regardless of route.

Subscription CLIs receive an allowlisted environment and run in temporary workspaces. K3 uses the documented Kimi Code membership API for headless and scheduled work, preserving Quorate's real client identity. The official CLI remains available only in an interactive terminal, runs without discovered skills, and receives its home and client-configuration paths because authentication requires them. Without a membership API key, non-interactive K3 calls fail closed. This containment limits ambient authority; it does not change the disclosure classification or replace review of Moonshot's current terms.

Moonshot's current terms allow submitted content to be used to operate, improve, and develop the service, including model improvement and research. A member can request an opt-out through `membership@moonshot.ai`. Until that opt-out is confirmed, and regardless of it afterward, K3 remains limited to public, synthetic, or deliberately sanitized Quorate inputs.

## Subscription renewal

A prepaid plan is a sunk cost. Before renewal, review the preceding 30 and 365 days of response-free telemetry, including K3 appearances, reachability, latency, route failures, and the API cost that comparable use would have incurred. Recheck the official price, entitlement, fair-use limits, regional restrictions, data-handling terms, and automatic-renewal rules.

Continue only when K3 remains admitted to at least one real role, usage is material enough to distinguish value from optionality, route reliability meets that role's floor, no unresolved term conflicts with the data boundary, and the plan price is justified by conservative API-equivalent cost or an explicitly documented strategic benefit. Otherwise, do not continue. Review the current monthly Allegretto plan at least seven days before each renewal and verify the actual billing date before taking account action.

## Current seat-four disposition

On 16 November 2026 Terry approved Gemini 4 Argon as a trial in seat four, replacing K3. K3 had refused every call since at least 1 October, after the Kimi membership was cancelled on 20 August. Argon passed the predeclared council-seat assay with `admit-trial` and all three fixed canaries through Antigravity without fallback diagnostics, as recorded in the [Gemini seat note](experiments/2026-10-29-gemini-seventh-seat-candidate.md). The seat supplies council, quick and brainstorm uses, as it did for K3. Promote it to production only after natural use supports its reliability and role value. Argon is not admitted as a judge, critic, coding route or mTOR role. The K3 history below stands as the record of that trial.

As of 17 July 2026, K3 is deployed as a trial in Quorate's Kimi family seat, which supplies its council and brainstorm uses. Its blind-phase role assay passed, and its membership API route passed all three fixed canaries without fallback diagnostics after the guarded entry point was restored. Its controlled downstream replay was inconclusive and response-free telemetry contains no natural K3 run yet. Interactive councils now collect categorical decision-value and K3-effect signals without retaining prose. Promote the seat to production status only after natural use supports its reliability and role value. K3 did not pass Vivesca's coding-route contract and is not admitted there. It is not admitted to mTOR; a non-coding challenger or reviewer role remains a candidate that requires its own gate.

## Fireworks US route candidate

As of 30 July 2026, Quorate has dormant client support for Fireworks'
`accounts/fireworks/routers/kimi-k3-us`, but the Kimi membership API remains the
default K3 route. The candidate accepts only Fireworks' Chat Completions endpoint
through its Priority serving tier and fails closed without falling through to
another model, provider, region, or serving tier. It is restricted to a member
seat, is rejected as a judge or critic, and does not participate in `--local`.

Fireworks documents the router as serving inference exclusively in the United
States and documents default zero data retention for open-model prompts and
generations. Those product controls do not broaden Quorate's disclosure
boundary. Prompt caching can retain content transiently in volatile memory,
request metadata is logged, and the reviewed public terms and DPA do not provide
an express blanket US-residency covenant. A regulated residency requirement
therefore needs a binding order form or MSA.

The Priority contract passed all three fixed canaries through `fireworks-us`
without diagnostics on 30 July 2026. The paired same-K3 assay then failed its
predeclared validity gates: the incumbent produced 20 of 20 operational passes,
while Fireworks produced 15 of 20 and truncated five responses. The resulting
15 eligible pairs left each judge with only 15 decisions, below the required
18. The route is not admitted to trial, the unchanged assay must not be rerun,
and the Kimi membership API remains the default. See the
[Fireworks US K3 candidate note](experiments/2026-07-29-fireworks-us-k3-route-candidate.md)
for the locked criteria and response-free evidence. Any reconsideration requires
a materially changed route or request contract, a newly predeclared assay, and
separate approval.
