# Fireworks US K3 route candidate

## Decision

Add a dormant, fail-closed client for Fireworks' US-only Kimi K3 router, but keep
`kimi-code/k3` as the fourth council seat. Do not start a Fireworks trial until a
dedicated credential exists, the same-K3 route assay passes, and the candidate
route itself clears all three production canaries.

## Evidence and boundary

Fireworks documents `accounts/fireworks/routers/kimi-k3-us` as serving inference
exclusively in the United States at a 10% premium. Its Chat Completions endpoint
uses an OpenAI-compatible request and returns final content separately from
`reasoning_content`. Fireworks also states that prompts and generations for open
models are not persistently logged by default, although request metadata is
logged and prompts may remain transiently in volatile prompt caches. The
Responses API is excluded because it stores conversations for up to 30 days by
default.

The relevant official sources are the
[US-only serverless route](https://docs.fireworks.ai/serverless/us-only-serverless),
[serverless serving paths](https://docs.fireworks.ai/serverless/serving-paths),
[Chat Completions contract](https://docs.fireworks.ai/api-reference/post-chatcompletions),
[data-handling policy](https://docs.fireworks.ai/guides/security_compliance/data_handling),
[prompt-caching policy](https://docs.fireworks.ai/guides/prompt-caching),
[privacy notice](https://fireworks.ai/privacy-policy), and
[terms of service](https://fireworks.ai/terms-of-service).

The US-only statement is a product-documentation commitment, not an express
blanket US-residency covenant in the reviewed public terms or DPA. Fireworks'
US domicile and default zero-data-retention posture therefore provide
defense in depth, not a broader data classification. Quorate remains limited
to public, synthetic, or deliberately sanitized inputs. If residency is to
operate as a regulated control, the order form or MSA must bind the exact
US-only router and permitted processing locations.

## Implemented candidate contract

The candidate client calls only
`POST https://api.fireworks.ai/inference/v1/chat/completions` with model
`accounts/fireworks/routers/kimi-k3-us`. Missing credentials, unsupported
Fireworks model IDs, transport failures, non-200 responses, malformed JSON, and
empty or truncated final content fail closed as the `fireworks-us` route. The
request makes context overflow an error rather than accepting Fireworks'
default silent truncation. The candidate is restricted to member-seat calls and
is rejected as a judge or critic. It never falls through to OpenRouter, Kimi, or
an ordinary Fireworks model.

After the live Standard-tier JSON canary timed out, the dormant client was
hardened to request `service_tier: "priority"` on every call. Fireworks describes
Priority as the serving path for higher reliability during peak traffic and
lists it as available for K3 US. This changes neither the model nor the US-only
router and does not introduce fallback; rejection or failure of Priority still
fails closed. It raises the published price to US$4.125 per million input tokens
and US$20.625 per million output tokens. Cached input remains conservatively
charged at the full input rate because Quorate does not retain cached-token
metadata.

The client deliberately omits `reasoning_effort`. Fireworks documents that
field generally but has not documented Kimi K3's accepted values or semantics;
the incumbent Kimi membership route also leaves effort at the model default.
Keeping the request controls aligned makes the first comparison a route test
rather than a route-and-inference-policy test. A reasoning-only response is
treated as an empty final answer.

The code resolves `QUORATE_FIREWORKS_KEY` before `FIREWORKS_API_KEY`, excludes
both names from subscription subprocess environments, records the distinct
`fireworks-us` provider and paid token usage, and recognizes `kimi-k3-us` for
K3-effect telemetry. Council debate calls are included in cost totals, and
usage snapshots attribute every member-seat and benchmark call's reachability,
latency, safe diagnostics, tokens, and estimated API cost by model and route.
They also expose a natural-use view that excludes synthetic benchmark calls.
The current estimate uses the Priority-tier prices above.

Saved benchmark evidence identifies the requested Priority contract as
`fireworks-us-k3-priority-v1`. The exact-route checker requires that marker and
`requested_service_tier: priority`; an older or unattested report cannot satisfy
the gate. These are client request attestations tied to the constants used to
construct the request, not provider-issued residency or serving-path
attestations.

A local response-free same-K3 assay harness now implements the paired
comparison. It preflights both candidate credentials before any call, uses the
fixed seed `20260729`, records all candidate attempts and eligible judgments
without response text, and refuses to overwrite an existing output. It records
the requested endpoints, models, Priority tier, and client contract. Any client
exception becomes sanitized failure evidence instead of aborting the report.

## Predeclared admission gate

1. Terry creates or selects the Fireworks account and API key. Add only its
   1Password reference to the guarded environment surfaces; do not put a
   credential in the repository.
2. Run the three fixed production canaries with
   `CONSILIUM_MODEL_M4=accounts/fireworks/routers/kimi-k3-us`. K3 itself must
   pass 3 of 3 through `fireworks-us`, with no diagnostics and the exact
   `fireworks-us-k3-priority-v1` request marker. Overall quorum health is not
   sufficient. Stop if this inexpensive Priority check fails.
3. Run the existing 20 synthetic council-seat cases as a paired, randomized,
   anonymized comparison of the incumbent `kimi-code/k3` route and the
   Fireworks US route. The prompts, output caps, timeout, and K3 default
   reasoning policy must be identical. Use the predeclared seed `20260729` and
   retain the single response-free output rather than rerunning for a favorable
   seed.
4. The run is valid only if the incumbent produces at least 18 reachable
   responses and each of the two judges produces at least 18 reachable,
   parseable decisions. Fireworks must produce at least 19 responses that are
   simultaneously reachable, output-compliant, and in time, use the exact
   `fireworks-us` client path throughout, and have median latency no more than
   twice the incumbent.
5. Treat a tie as 0.5 Fireworks points, a Fireworks preference as 1, and an
   incumbent preference as 0 across the 40 possible judgments. Require at
   least 18 points, a predeclared 10% non-inferiority margin around the
   20-point equal-route expectation, plus every operational floor.
6. Only then begin a reversible M4 trial. Promote the route to production only
   after response-free natural-use telemetry supports reachability, latency,
   cost, and K3's decision value.

The approved operator should run each live gate once. The Priority canary
artifact is
`/Users/terry/tmp/quorate-fireworks-k3-priority-v1-canary.json`;
create and validate it with:

```bash
canary_artifact=/Users/terry/tmp/quorate-fireworks-k3-priority-v1-canary.json
canary_error=/Users/terry/tmp/quorate-fireworks-k3-priority-v1-canary.err
[[ ! -e "$canary_artifact" && ! -e "$canary_error" ]] || exit 1
set -o noclobber
CONSILIUM_MODEL_M4=accounts/fireworks/routers/kimi-k3-us \
  /Users/terry/.local/bin/quorate benchmark --timeout 90 --no-save --json \
  > "$canary_artifact" 2> "$canary_error"
/usr/bin/python3 scripts/check_fireworks_benchmark.py --required \
  "$canary_artifact"
```

Only an attested 3-of-3 pass permits the paired assay:

```bash
uv run python -m quorate.k3_route_assay --execute-live \
  --output /Users/terry/tmp/quorate-k3-route-assay-priority-v1-seed-20260729.json
```

Its implicit seed is the fixed seed above, it makes at most 80 synthetic model
calls, and the output path cannot already exist. Retain both response-free
artifacts, including a failed result, rather than selecting among reruns.

## Live validation

On 29 July 2026, the dedicated credential was stored by reference in the
guarded environment and the fixed response-free benchmark was run with
`CONSILIUM_MODEL_M4=accounts/fireworks/routers/kimi-k3-us`. An initial HTTP 401
run exposed a credential-preservation mismatch between the one-time key and the
protected item. The item was corrected from the still-open one-time display and
verified against the resolved cache by hash before one clean rerun. The invalid
run is harness evidence, not route evidence.

The clean rerun authenticated and retained overall quorum, but the Fireworks
seat passed only 2 of 3 canaries. The exact-token and simple-reasoning canaries
passed through `fireworks-us` in 2.56 and 3.39 seconds. The JSON-contract call
timed out after 180.15 seconds and therefore recorded provider `none` with
diagnostic `fireworks-us:timeout`. The seat summary was 2 reachable, 2 passed,
providers `{"fireworks-us": 2, "none": 1}`, and an estimated API cost of
US$0.0030.

This Standard-tier result failed the predeclared 3-of-3 route gate.

On 30 July 2026, one separately approved Priority-tier run used the
`fireworks-us-k3-priority-v1` client contract. K3 passed all three canaries
through `fireworks-us` with no diagnostics: exact token in 4.08 seconds, JSON
contract in 15.00 seconds and simple reasoning in 11.38 seconds. The exact-route
checker exited 0, overall quorum remained healthy, and the K3 calls used 335
input and 409 output tokens at an estimated US$0.0098; total estimated benchmark
cost was US$0.0118.

The read-only response-free artifact is
`/Users/terry/tmp/quorate-fireworks-k3-priority-v1-canary.json`, SHA-256
`e8ce8fb4c4c026d7346f193a6b7c4570e1cf9d5c662a151048a68795694810fa`.
Its stderr artifact is empty. This clears the Priority 3-of-3 canary gate once;
it does not admit or activate the route. The incumbent remains the default and
the paired 20-case route assay has not been run.

## Unresolved gates

The single predeclared paired same-K3 route assay is now the next admission
gate. It requires separate approval because it makes at most 80 synthetic model
calls. Only a valid assay pass may begin a reversible M4 override trial; the
incumbent remains the default meanwhile. K3-specific reasoning controls and the
effect of transient prompt caching remain live-test questions. A binding
contractual US-residency commitment remains a separate procurement gate for
regulated or sensitive use.
