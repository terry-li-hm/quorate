"""Response-free same-K3 route admission assay.

This module compares K3's incumbent membership API with the dormant Fireworks
US route. It deliberately reuses the fixed council-seat cases without changing
the existing K3-versus-K2.6 assay or its promotion semantics.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import httpx

from quorate.api import (
    _diagnostic_code,
    _fireworks_us_api,
    _kimi_code_api,
    run_parallel,
)
from quorate.config import (
    FIREWORKS_URL,
    FIREWORKS_US_CLIENT_CONTRACT,
    FIREWORKS_US_K3_MODEL,
    FIREWORKS_US_SERVICE_TIER,
    KIMI_CODE_URL,
    Message,
    ModelCallResult,
    ModelEntry,
    ReasoningEffort,
    api_keys,
    is_error,
)
from quorate.council_seat_assay import (
    BLIND_SYSTEM,
    CASES,
    FABLE,
    GPT,
    JUDGE_SYSTEM,
    Case,
    output_compliant,
    pairwise_prompt,
    parse_pairwise,
    word_count,
)
from quorate.council_seat_assay import (
    SUITE_VERSION as CASE_SUITE_VERSION,
)

SUITE_VERSION = "2026-07-29"
MAX_TOKENS = 1024
DEFAULT_TIMEOUT_S = 300.0
JUDGE_EFFORT = ReasoningEffort.HIGH
FIREWORKS_POINTS_FLOOR = 18.0
POSSIBLE_JUDGMENTS = len(CASES) * 2
PREDECLARED_SEED = 20260729


@dataclass(frozen=True)
class RouteSpec:
    id: str
    entry: ModelEntry
    expected_provider: str


INCUMBENT = RouteSpec(
    "incumbent",
    ModelEntry("Kimi-K3 membership API", "kimi-code/k3"),
    "kimi-code-api",
)
FIREWORKS = RouteSpec(
    "fireworks-us",
    ModelEntry("Kimi-K3 Fireworks US", FIREWORKS_US_K3_MODEL),
    "fireworks-us",
)
ROUTES = (INCUMBENT, FIREWORKS)
JUDGES = (FABLE, GPT)

CandidatePair = tuple[ModelCallResult, ModelCallResult]
CandidateRunner = Callable[[Case, float], Awaitable[CandidatePair]]
JudgeRunner = Callable[
    [ModelEntry, list[Message], int, float, ReasoningEffort],
    Awaitable[ModelCallResult],
]
OrderPlan = dict[tuple[str, str], bool]


class CandidateCredentialError(RuntimeError):
    """Raised before live work when either exact candidate credential is absent."""


def _require_candidate_credentials() -> dict[str, str | None]:
    keys = api_keys()
    if not keys.get("kimi_code") or not keys.get("fireworks"):
        raise CandidateCredentialError(
            "Both KIMI_CODE_API_KEY and a Fireworks API key are required; "
            "no candidate call was made"
        )
    return keys


def suite_fingerprint() -> str:
    """Hash the fixed prompt corpus and hidden judging criteria."""
    payload = {
        "blind_system": BLIND_SYSTEM,
        "judge_system": JUDGE_SYSTEM,
        "cases": [asdict(case) for case in CASES],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_order_plan(seed: int) -> OrderPlan:
    """Randomize A/B labels while balancing every case and both judges."""
    if len(CASES) % 2:
        raise ValueError("The fixed case suite must have an even number of cases")
    fireworks_is_a = [True] * (len(CASES) // 2) + [False] * (len(CASES) // 2)
    random.Random(seed).shuffle(fireworks_is_a)
    plan: OrderPlan = {}
    for case, fable_order in zip(CASES, fireworks_is_a, strict=True):
        plan[(case.id, FABLE.name)] = fable_order
        plan[(case.id, GPT.name)] = not fable_order
    return plan


def summarize_order_plan(plan: OrderPlan, seed: int) -> dict[str, Any]:
    """Return response-free evidence that the anonymous order is balanced."""
    by_judge = {
        judge.name: {
            "fireworks_as_a": sum(plan[(case.id, judge.name)] for case in CASES),
            "fireworks_as_b": sum(not plan[(case.id, judge.name)] for case in CASES),
        }
        for judge in JUDGES
    }
    opposite_per_case = all(
        plan[(case.id, FABLE.name)] != plan[(case.id, GPT.name)] for case in CASES
    )
    balanced = opposite_per_case and all(
        counts["fireworks_as_a"] == len(CASES) // 2 and counts["fireworks_as_b"] == len(CASES) // 2
        for counts in by_judge.values()
    )
    return {
        "strategy": "seeded-random-balanced-crossover",
        "seed": seed,
        "by_judge": by_judge,
        "opposite_per_case": opposite_per_case,
        "balanced": balanced,
    }


def _failed_route_result(spec: RouteSpec, diagnostic: str, start: float) -> ModelCallResult:
    return ModelCallResult(
        name=spec.entry.name,
        model_id=spec.entry.model,
        response=f"[Error: {spec.id} route failed]",
        provider="none",
        latency_s=time.monotonic() - start,
        diagnostics=(f"{spec.expected_provider}:{diagnostic}",),
    )


def _route_result(
    spec: RouteSpec,
    content: str,
    tokens: dict[str, int] | None,
    start: float,
) -> ModelCallResult:
    if is_error(content):
        return _failed_route_result(spec, _diagnostic_code(content), start)
    return ModelCallResult(
        name=spec.entry.name,
        model_id=spec.entry.model,
        response=content,
        provider=spec.expected_provider,
        latency_s=time.monotonic() - start,
        tokens_in=tokens.get("tokens_in") if tokens else None,
        tokens_out=tokens.get("tokens_out") if tokens else None,
    )


async def _query_incumbent(
    client: httpx.AsyncClient,
    keys: dict[str, str | None],
    messages: list[Message],
    timeout: float,
) -> ModelCallResult:
    start = time.monotonic()
    key = keys.get("kimi_code")
    if not key:
        return _failed_route_result(INCUMBENT, "no_credentials", start)
    try:
        content, tokens = await _kimi_code_api(
            client,
            key,
            INCUMBENT.entry.model,
            messages,
            MAX_TOKENS,
            timeout,
        )
    except Exception:
        return _failed_route_result(INCUMBENT, "client_exception", start)
    return _route_result(INCUMBENT, content, tokens, start)


async def _query_fireworks(
    client: httpx.AsyncClient,
    keys: dict[str, str | None],
    messages: list[Message],
    timeout: float,
) -> ModelCallResult:
    start = time.monotonic()
    key = keys.get("fireworks")
    if not key:
        return _failed_route_result(FIREWORKS, "no_credentials", start)
    try:
        content, tokens = await _fireworks_us_api(
            client,
            key,
            FIREWORKS.entry.model,
            messages,
            MAX_TOKENS,
            timeout,
            None,
        )
    except Exception:
        return _failed_route_result(FIREWORKS, "client_exception", start)
    return _route_result(FIREWORKS, content, tokens, start)


async def query_candidate_pair(
    case: Case,
    timeout: float,
    *,
    keys: dict[str, str | None] | None = None,
) -> CandidatePair:
    """Call only the two exact K3 APIs with one identical request contract."""
    resolved_keys = api_keys() if keys is None else keys
    missing_incumbent = not resolved_keys.get("kimi_code")
    missing_fireworks = not resolved_keys.get("fireworks")
    if missing_incumbent or missing_fireworks:
        start = time.monotonic()
        return (
            _failed_route_result(
                INCUMBENT,
                "no_credentials" if missing_incumbent else "credential_preflight_failed",
                start,
            ),
            _failed_route_result(
                FIREWORKS,
                "no_credentials" if missing_fireworks else "credential_preflight_failed",
                start,
            ),
        )
    messages = [Message.system(BLIND_SYSTEM), Message.user(case.question)]
    async with httpx.AsyncClient() as client:
        incumbent, fireworks = await asyncio.gather(
            _query_incumbent(client, resolved_keys, messages, timeout),
            _query_fireworks(client, resolved_keys, messages, timeout),
        )
    return incumbent, fireworks


async def query_judge(
    judge: ModelEntry,
    messages: list[Message],
    max_tokens: int,
    timeout: float,
    effort: ReasoningEffort,
) -> ModelCallResult:
    """Query one of the two locked judges through Quorate's normal route."""
    return (
        await run_parallel(
            [judge],
            messages,
            max_tokens=max_tokens,
            timeout=timeout,
            effort=effort,
        )
    )[0]


def _route_isolated(result: ModelCallResult, spec: RouteSpec) -> bool:
    if result.model_id != spec.entry.model:
        return False
    if not result.is_error:
        return result.provider == spec.expected_provider and not result.diagnostics
    return (
        result.provider == "none"
        and bool(result.diagnostics)
        and all(
            diagnostic.startswith(f"{spec.expected_provider}:") for diagnostic in result.diagnostics
        )
    )


def candidate_attempt(
    case: Case,
    spec: RouteSpec,
    result: ModelCallResult,
    timeout: float,
) -> dict[str, Any]:
    """Reduce an ephemeral response to response-free route telemetry."""
    isolated = _route_isolated(result, spec)
    reachable = not result.is_error and result.provider == spec.expected_provider
    compliant = reachable and output_compliant(result.response)
    within_timeout = reachable and result.latency_s <= timeout
    words = word_count(result.response) if reachable else 0
    return {
        "case": case.id,
        "domain": case.domain,
        "route": spec.id,
        "requested_model_id": spec.entry.model,
        "result_model_id": result.model_id,
        "provider": result.provider,
        "reachable": reachable,
        "output_compliant": compliant,
        "within_timeout": within_timeout,
        "operational_pass": compliant and within_timeout,
        "route_isolated": isolated,
        "latency_s": result.latency_s,
        "words": words,
        "response_chars": len(result.response) if reachable else 0,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "diagnostics": list(result.diagnostics),
    }


def summarize_route(spec: RouteSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    route_rows = [row for row in rows if row["route"] == spec.id]
    latencies = [float(row["latency_s"]) for row in route_rows if row["reachable"]]
    diagnostics = Counter(
        diagnostic for row in route_rows for diagnostic in row.get("diagnostics", [])
    )
    return {
        "route": spec.id,
        "requested_model_id": spec.entry.model,
        "expected_provider": spec.expected_provider,
        "cases": len(route_rows),
        "reachable": sum(bool(row["reachable"]) for row in route_rows),
        "output_compliant": sum(bool(row["output_compliant"]) for row in route_rows),
        "within_timeout": sum(bool(row["within_timeout"]) for row in route_rows),
        "operational_pass": sum(bool(row["operational_pass"]) for row in route_rows),
        "route_isolated": len(route_rows) == len(CASES)
        and all(bool(row["route_isolated"]) for row in route_rows),
        "mean_latency_s": round(mean(latencies), 3) if latencies else None,
        "median_latency_s": round(median(latencies), 3) if latencies else None,
        "providers": dict(sorted(Counter(str(row["provider"]) for row in route_rows).items())),
        "diagnostics": dict(sorted(diagnostics.items())),
    }


def judge_attempt(
    case: Case,
    judge: ModelCallResult,
    *,
    fireworks_is_a: bool,
) -> dict[str, Any]:
    """Reduce a judge response to its anonymous decision and safe telemetry."""
    decision = parse_pairwise(judge.response) if not judge.is_error else None
    a_route = FIREWORKS.id if fireworks_is_a else INCUMBENT.id
    b_route = INCUMBENT.id if fireworks_is_a else FIREWORKS.id
    winner = {"A": a_route, "B": b_route, "TIE": "tie"}.get(decision or "")
    points = 0.5 if winner == "tie" else 1.0 if winner == FIREWORKS.id else 0.0
    return {
        "case": case.id,
        "judge": judge.name,
        "judge_model_id": judge.model_id,
        "provider": judge.provider,
        "reachable": not judge.is_error,
        "valid_decision": decision is not None,
        "decision": decision,
        "a_route": a_route,
        "b_route": b_route,
        "winner": winner,
        "fireworks_points": points if decision is not None else None,
        "latency_s": judge.latency_s,
        "response_chars": len(judge.response) if not judge.is_error else 0,
        "diagnostics": list(judge.diagnostics),
    }


async def judge_case(
    case: Case,
    pair: CandidatePair,
    order_plan: OrderPlan,
    semaphore: asyncio.Semaphore,
    timeout: float,
    runner: JudgeRunner,
) -> list[dict[str, Any]]:
    """Judge one eligible candidate pair with opposite anonymous orderings."""
    by_route = {spec.id: result for spec, result in zip(ROUTES, pair, strict=True)}

    async def evaluate(judge: ModelEntry) -> dict[str, Any]:
        fireworks_is_a = order_plan[(case.id, judge.name)]
        a_response = (
            by_route[FIREWORKS.id].response if fireworks_is_a else by_route[INCUMBENT.id].response
        )
        b_response = (
            by_route[INCUMBENT.id].response if fireworks_is_a else by_route[FIREWORKS.id].response
        )
        messages = [
            Message.system(JUDGE_SYSTEM),
            Message.user(pairwise_prompt(case, a_response, b_response)),
        ]
        async with semaphore:
            result = await runner(judge, messages, MAX_TOKENS, timeout, JUDGE_EFFORT)
        return judge_attempt(case, result, fireworks_is_a=fireworks_is_a)

    return list(await asyncio.gather(*(evaluate(judge) for judge in JUDGES)))


def summarize_judges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for judge in JUDGES:
        judge_rows = [row for row in rows if row["judge"] == judge.name]
        points = [
            float(row["fireworks_points"])
            for row in judge_rows
            if row["fireworks_points"] is not None
        ]
        summaries.append(
            {
                "judge": judge.name,
                "model_id": judge.model,
                "cases": len(judge_rows),
                "reachable": sum(bool(row["reachable"]) for row in judge_rows),
                "valid_decisions": sum(bool(row["valid_decision"]) for row in judge_rows),
                "fireworks_points": round(sum(points), 1),
                "fireworks_wins": sum(row["winner"] == FIREWORKS.id for row in judge_rows),
                "incumbent_wins": sum(row["winner"] == INCUMBENT.id for row in judge_rows),
                "ties": sum(row["winner"] == "tie" for row in judge_rows),
            }
        )
    return summaries


def admission_decision(
    route_summaries: list[dict[str, Any]],
    judge_summaries: list[dict[str, Any]],
    ordering: dict[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared route-validity and non-inferiority gates."""
    by_route = {row["route"]: row for row in route_summaries}
    by_judge = {row["judge"]: row for row in judge_summaries}
    incumbent = by_route[INCUMBENT.id]
    fireworks = by_route[FIREWORKS.id]
    incumbent_median = float(incumbent["median_latency_s"] or 0)
    fireworks_median = float(fireworks["median_latency_s"] or 0)
    fireworks_points = round(
        sum(float(row["fireworks_points"]) for row in judge_summaries),
        1,
    )
    validity_gates = {
        "fixed_twenty_case_suite": int(incumbent["cases"]) == len(CASES)
        and int(fireworks["cases"]) == len(CASES),
        "randomized_balanced_anonymous_order": bool(ordering["balanced"]),
        "incumbent_membership_api_only": bool(incumbent["route_isolated"]),
        "incumbent_reachable_at_least_18": int(incumbent["reachable"]) >= 18,
        "fireworks_us_route_only": bool(fireworks["route_isolated"]),
        "fireworks_reachable_at_least_19": int(fireworks["reachable"]) >= 19,
        "fireworks_output_compliant_at_least_19": int(fireworks["output_compliant"]) >= 19,
        "fireworks_in_time_at_least_19": int(fireworks["within_timeout"]) >= 19,
        "fireworks_compound_operational_pass_at_least_19": int(fireworks["operational_pass"]) >= 19,
        "fireworks_median_latency_at_most_2x_incumbent": incumbent_median > 0
        and fireworks_median <= 2 * incumbent_median,
        "fable_reachable_at_least_18": int(by_judge[FABLE.name]["reachable"]) >= 18,
        "fable_valid_decisions_at_least_18": int(by_judge[FABLE.name]["valid_decisions"]) >= 18,
        "gpt_reachable_at_least_18": int(by_judge[GPT.name]["reachable"]) >= 18,
        "gpt_valid_decisions_at_least_18": int(by_judge[GPT.name]["valid_decisions"]) >= 18,
    }
    quality_gate = {
        "fireworks_points_at_least_18_of_40": fireworks_points >= FIREWORKS_POINTS_FLOOR
    }
    valid_run = all(validity_gates.values())
    if not valid_run:
        action = "invalid-run"
        reason = "At least one non-compensable route or assay validity gate failed."
    elif not all(quality_gate.values()):
        action = "retain-incumbent"
        reason = "The valid comparison did not clear the predeclared quality floor."
    else:
        action = "eligible-for-3-of-3-canary"
        reason = "Every route gate passed; the exact-route canary remains the next gate."
    return {
        "action": action,
        "valid_run": valid_run,
        "fireworks_points": fireworks_points,
        "possible_points": POSSIBLE_JUDGMENTS,
        "validity_gates": validity_gates,
        "quality_gate": quality_gate,
        "reason": reason,
    }


def _pair_eligible(pair: CandidatePair) -> bool:
    return all(
        not result.is_error and result.provider == spec.expected_provider and not result.diagnostics
        for spec, result in zip(ROUTES, pair, strict=True)
    )


async def run_assay(
    *,
    concurrency: int = 4,
    timeout: float = DEFAULT_TIMEOUT_S,
    seed: int = PREDECLARED_SEED,
    candidate_runner: CandidateRunner | None = None,
    judge_runner: JudgeRunner | None = None,
) -> dict[str, Any]:
    """Run the paired assay and return a response-free aggregate report."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    order_plan = build_order_plan(seed)
    ordering = summarize_order_plan(order_plan, seed)
    if candidate_runner is None:
        candidate_keys = _require_candidate_credentials()

        async def candidate_call(case: Case, candidate_timeout: float) -> CandidatePair:
            return await query_candidate_pair(
                case,
                candidate_timeout,
                keys=candidate_keys,
            )

    else:
        candidate_call = candidate_runner
    judge_call = judge_runner or query_judge
    candidate_semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_candidate(case: Case) -> CandidatePair:
        async with candidate_semaphore:
            return await candidate_call(case, timeout)

    candidate_pairs = await asyncio.gather(*(evaluate_candidate(case) for case in CASES))
    candidate_rows = [
        candidate_attempt(case, spec, result, timeout)
        for case, pair in zip(CASES, candidate_pairs, strict=True)
        for spec, result in zip(ROUTES, pair, strict=True)
    ]
    judge_semaphore = asyncio.Semaphore(concurrency)
    judged_cases = await asyncio.gather(
        *(
            judge_case(
                case,
                pair,
                order_plan,
                judge_semaphore,
                timeout,
                judge_call,
            )
            for case, pair in zip(CASES, candidate_pairs, strict=True)
            if _pair_eligible(pair)
        )
    )
    judgment_rows = [row for case_rows in judged_cases for row in case_rows]
    route_summaries = [summarize_route(spec, candidate_rows) for spec in ROUTES]
    judge_summaries = summarize_judges(judgment_rows)
    decision = admission_decision(route_summaries, judge_summaries, ordering)
    return {
        "schema_version": 1,
        "suite_version": SUITE_VERSION,
        "case_suite_version": CASE_SUITE_VERSION,
        "suite_fingerprint": suite_fingerprint(),
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cases": len(CASES),
        "candidate_concurrency": concurrency,
        "request_contract": {
            "prompt": "fixed blind system plus fixed case question",
            "max_tokens": MAX_TOKENS,
            "timeout_s": timeout,
            "candidate_reasoning_policy": "provider-default",
            "identical_for_both_routes": True,
            "incumbent_endpoint": KIMI_CODE_URL,
            "incumbent_requested_model": INCUMBENT.entry.model,
            "fireworks_endpoint": FIREWORKS_URL,
            "fireworks_requested_model": FIREWORKS.entry.model,
            "fireworks_client_contract": FIREWORKS_US_CLIENT_CONTRACT,
            "fireworks_requested_service_tier": FIREWORKS_US_SERVICE_TIER,
        },
        "judges": {
            "models": [judge.model for judge in JUDGES],
            "max_tokens": MAX_TOKENS,
            "timeout_s": timeout,
            "reasoning_effort": JUDGE_EFFORT.value,
        },
        "ordering": ordering,
        "route_summaries": route_summaries,
        "judge_summaries": judge_summaries,
        "decision": decision,
        "candidate_attempts": candidate_rows,
        "judgments": judgment_rows,
        "responses_retained": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the live, response-free same-K3 route admission assay."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--execute-live",
        action="store_true",
        help="Required acknowledgement that this command makes external model calls.",
    )
    arguments = parser.parse_args()
    if not arguments.execute_live:
        parser.error("--execute-live is required; no provider call was made")
    if arguments.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.exists():
        parser.error("output already exists; no provider call was made")
    try:
        report = asyncio.run(
            run_assay(
                concurrency=arguments.concurrency,
                timeout=arguments.timeout,
            )
        )
    except CandidateCredentialError as exc:
        parser.error(str(exc))
    try:
        with arguments.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError:
        parser.error("output was created concurrently and was not overwritten")
    print(
        json.dumps(
            {"output": str(arguments.output), **report["decision"]},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["decision"]["action"] == "eligible-for-3-of-3-canary" else 1


if __name__ == "__main__":
    raise SystemExit(main())
