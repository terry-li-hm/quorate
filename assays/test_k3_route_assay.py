import asyncio
import copy
import json

import pytest

import quorate.k3_route_assay as assay
from quorate.config import ModelCallResult
from quorate.council_seat_assay import CASES as COUNCIL_SEAT_CASES


def _words(label: str) -> str:
    return " ".join([label, *(["word"] * 59)])


def _candidate_result(
    spec: assay.RouteSpec,
    *,
    response: str | None = None,
    provider: str | None = None,
    latency: float = 10.0,
    diagnostics: tuple[str, ...] = (),
) -> ModelCallResult:
    return ModelCallResult(
        name=spec.entry.name,
        model_id=spec.entry.model,
        response=response or _words(spec.id),
        provider=provider or spec.expected_provider,
        latency_s=latency,
        diagnostics=diagnostics,
    )


def _passing_route_summaries() -> list[dict]:
    return [
        {
            "route": assay.INCUMBENT.id,
            "cases": 20,
            "reachable": 18,
            "output_compliant": 18,
            "within_timeout": 18,
            "operational_pass": 18,
            "route_isolated": True,
            "median_latency_s": 10.0,
        },
        {
            "route": assay.FIREWORKS.id,
            "cases": 20,
            "reachable": 19,
            "output_compliant": 19,
            "within_timeout": 19,
            "operational_pass": 19,
            "route_isolated": True,
            "median_latency_s": 20.0,
        },
    ]


def _passing_judge_summaries(points: float = 9.0) -> list[dict]:
    return [
        {
            "judge": assay.FABLE.name,
            "reachable": 18,
            "valid_decisions": 18,
            "fireworks_points": points,
        },
        {
            "judge": assay.GPT.name,
            "reachable": 18,
            "valid_decisions": 18,
            "fireworks_points": points,
        },
    ]


def test_assay_reuses_the_locked_twenty_case_suite():
    assert assay.CASES is COUNCIL_SEAT_CASES
    assert len(assay.CASES) == 20
    assert len({case.id for case in assay.CASES}) == 20


def test_order_plan_is_seeded_anonymous_balanced_and_crossed():
    first = assay.build_order_plan(129)
    second = assay.build_order_plan(129)
    summary = assay.summarize_order_plan(first, 129)

    assert first == second
    assert summary["balanced"] is True
    assert summary["opposite_per_case"] is True
    assert summary["by_judge"] == {
        assay.FABLE.name: {"fireworks_as_a": 10, "fireworks_as_b": 10},
        assay.GPT.name: {"fireworks_as_a": 10, "fireworks_as_b": 10},
    }
    for case in assay.CASES:
        assert first[(case.id, assay.FABLE.name)] != first[(case.id, assay.GPT.name)]


def test_candidate_routes_receive_identical_locked_request_contract(monkeypatch):
    captured: dict[str, dict] = {}
    monkeypatch.setattr(
        assay,
        "api_keys",
        lambda: {"kimi_code": "membership-key", "fireworks": "fireworks-key"},
    )

    async def fake_kimi(
        _client,
        key,
        model,
        messages,
        max_tokens,
        timeout,
    ):
        captured["incumbent"] = {
            "key": key,
            "model": model,
            "messages": [message.to_dict() for message in messages],
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        return _words("incumbent"), {"tokens_in": 10, "tokens_out": 20}

    async def fake_fireworks(
        _client,
        key,
        model,
        messages,
        max_tokens,
        timeout,
        effort,
    ):
        captured["fireworks"] = {
            "key": key,
            "model": model,
            "messages": [message.to_dict() for message in messages],
            "max_tokens": max_tokens,
            "timeout": timeout,
            "effort": effort,
        }
        return _words("fireworks"), {"tokens_in": 10, "tokens_out": 20}

    monkeypatch.setattr(assay, "_kimi_code_api", fake_kimi)
    monkeypatch.setattr(assay, "_fireworks_us_api", fake_fireworks)

    incumbent, fireworks = asyncio.run(assay.query_candidate_pair(assay.CASES[0], 47.0))

    assert incumbent.provider == "kimi-code-api"
    assert fireworks.provider == "fireworks-us"
    assert captured["incumbent"]["messages"] == captured["fireworks"]["messages"]
    assert captured["incumbent"]["max_tokens"] == captured["fireworks"]["max_tokens"] == 1024
    assert captured["incumbent"]["timeout"] == captured["fireworks"]["timeout"] == 47.0
    assert captured["fireworks"]["effort"] is None
    assert captured["incumbent"]["model"] == "kimi-code/k3"
    assert captured["fireworks"]["model"] == "accounts/fireworks/routers/kimi-k3-us"


def test_missing_candidate_credentials_fail_closed_without_a_provider_call(monkeypatch):
    monkeypatch.setattr(assay, "api_keys", lambda: {})

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("No provider call is allowed without its exact credential")

    monkeypatch.setattr(assay, "_kimi_code_api", fail_if_called)
    monkeypatch.setattr(assay, "_fireworks_us_api", fail_if_called)

    incumbent, fireworks = asyncio.run(assay.query_candidate_pair(assay.CASES[0], 47.0))

    assert incumbent.is_error
    assert incumbent.provider == "none"
    assert incumbent.diagnostics == ("kimi-code-api:no_credentials",)
    assert fireworks.is_error
    assert fireworks.provider == "none"
    assert fireworks.diagnostics == ("fireworks-us:no_credentials",)


def test_partial_candidate_credentials_preflight_before_any_live_call(monkeypatch):
    monkeypatch.setattr(
        assay,
        "api_keys",
        lambda: {"kimi_code": "membership-key", "fireworks": None},
    )

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Neither provider may be called after a partial preflight")

    monkeypatch.setattr(assay, "_kimi_code_api", fail_if_called)
    monkeypatch.setattr(assay, "_fireworks_us_api", fail_if_called)

    incumbent, fireworks = asyncio.run(assay.query_candidate_pair(assay.CASES[0], 47.0))

    assert incumbent.diagnostics == ("kimi-code-api:credential_preflight_failed",)
    assert fireworks.diagnostics == ("fireworks-us:no_credentials",)
    with pytest.raises(assay.CandidateCredentialError, match="no candidate call was made"):
        asyncio.run(assay.run_assay())


def test_malformed_incumbent_response_becomes_safe_failure_evidence(monkeypatch):
    monkeypatch.setattr(
        assay,
        "api_keys",
        lambda: {"kimi_code": "membership-key", "fireworks": "fireworks-key"},
    )

    async def malformed_incumbent(*_args, **_kwargs):
        raise ValueError("provider prose that must not be retained")

    async def healthy_fireworks(*_args, **_kwargs):
        return _words("fireworks"), {"tokens_in": 10, "tokens_out": 20}

    monkeypatch.setattr(assay, "_kimi_code_api", malformed_incumbent)
    monkeypatch.setattr(assay, "_fireworks_us_api", healthy_fireworks)

    incumbent, fireworks = asyncio.run(assay.query_candidate_pair(assay.CASES[0], 47.0))

    assert incumbent.is_error
    assert incumbent.diagnostics == ("kimi-code-api:client_exception",)
    assert "provider prose" not in incumbent.response
    assert fireworks.provider == "fireworks-us"
    assert not fireworks.is_error


def test_wrong_result_model_id_fails_route_isolation():
    result = _candidate_result(assay.FIREWORKS)
    result.model_id = "different/model"

    row = assay.candidate_attempt(
        assay.CASES[0],
        assay.FIREWORKS,
        result,
        47.0,
    )

    assert row["requested_model_id"] == "accounts/fireworks/routers/kimi-k3-us"
    assert row["result_model_id"] == "different/model"
    assert row["route_isolated"] is False


def test_exact_gate_floor_advances_only_to_the_canary():
    ordering = assay.summarize_order_plan(assay.build_order_plan(129), 129)
    decision = assay.admission_decision(
        _passing_route_summaries(),
        _passing_judge_summaries(),
        ordering,
    )

    assert decision["action"] == "eligible-for-3-of-3-canary"
    assert decision["valid_run"] is True
    assert decision["fireworks_points"] == 18.0
    assert all(decision["validity_gates"].values())
    assert all(decision["quality_gate"].values())


@pytest.mark.parametrize(
    ("section", "index", "field", "value"),
    [
        ("route", 0, "reachable", 17),
        ("route", 0, "route_isolated", False),
        ("route", 1, "reachable", 18),
        ("route", 1, "output_compliant", 18),
        ("route", 1, "within_timeout", 18),
        ("route", 1, "operational_pass", 18),
        ("route", 1, "route_isolated", False),
        ("route", 1, "median_latency_s", 20.001),
        ("judge", 0, "reachable", 17),
        ("judge", 0, "valid_decisions", 17),
        ("judge", 1, "reachable", 17),
        ("judge", 1, "valid_decisions", 17),
    ],
)
def test_each_non_compensable_floor_invalidates_the_run(section, index, field, value):
    routes = _passing_route_summaries()
    judges = _passing_judge_summaries()
    target = routes if section == "route" else judges
    target[index][field] = value
    ordering = assay.summarize_order_plan(assay.build_order_plan(129), 129)

    decision = assay.admission_decision(routes, judges, ordering)

    assert decision["action"] == "invalid-run"
    assert decision["valid_run"] is False


def test_unbalanced_order_invalidates_the_run():
    ordering = assay.summarize_order_plan(assay.build_order_plan(129), 129)
    ordering["balanced"] = False

    decision = assay.admission_decision(
        _passing_route_summaries(),
        _passing_judge_summaries(),
        ordering,
    )

    assert decision["action"] == "invalid-run"


def test_disjoint_fireworks_failures_do_not_satisfy_compound_operational_floor():
    rows = []
    for index, case in enumerate(assay.CASES):
        response = _words("fireworks")
        latency = 10.0
        if index == 0:
            response = " ".join(["overlong"] * 221)
        if index == 1:
            latency = 48.0
        rows.append(
            assay.candidate_attempt(
                case,
                assay.FIREWORKS,
                _candidate_result(
                    assay.FIREWORKS,
                    response=response,
                    latency=latency,
                ),
                47.0,
            )
        )
    fireworks = assay.summarize_route(assay.FIREWORKS, rows)
    routes = _passing_route_summaries()
    routes[1] = fireworks
    ordering = assay.summarize_order_plan(assay.build_order_plan(129), 129)

    decision = assay.admission_decision(routes, _passing_judge_summaries(), ordering)

    assert fireworks["reachable"] == 20
    assert fireworks["output_compliant"] == 19
    assert fireworks["within_timeout"] == 19
    assert fireworks["operational_pass"] == 18
    assert decision["action"] == "invalid-run"
    assert decision["validity_gates"]["fireworks_compound_operational_pass_at_least_19"] is False


def test_valid_run_below_eighteen_points_retains_incumbent():
    ordering = assay.summarize_order_plan(assay.build_order_plan(129), 129)
    decision = assay.admission_decision(
        _passing_route_summaries(),
        _passing_judge_summaries(points=8.75),
        ordering,
    )

    assert decision["action"] == "retain-incumbent"
    assert decision["valid_run"] is True
    assert decision["fireworks_points"] == 17.5
    assert decision["quality_gate"] == {"fireworks_points_at_least_18_of_40": False}


def test_assay_report_retains_no_candidate_or_judge_response_text():
    candidate_text: set[str] = set()
    judge_text: set[str] = set()

    async def fake_candidates(case, _timeout):
        incumbent_response = _words(f"alpha-{case.id}")
        fireworks_response = _words(f"beta-{case.id}")
        candidate_text.update((incumbent_response, fireworks_response))
        return (
            _candidate_result(
                assay.INCUMBENT,
                response=incumbent_response,
                latency=10.0,
            ),
            _candidate_result(
                assay.FIREWORKS,
                response=fireworks_response,
                latency=15.0,
            ),
        )

    async def fake_judge(judge, messages, max_tokens, timeout, effort):
        assert max_tokens == assay.MAX_TOKENS
        assert timeout == 47.0
        assert effort is assay.JUDGE_EFFORT
        prompt = messages[-1].content
        assert assay.INCUMBENT.entry.name not in prompt
        assert assay.FIREWORKS.entry.name not in prompt
        assert assay.INCUMBENT.entry.model not in prompt
        assert assay.FIREWORKS.entry.model not in prompt
        decision = "A" if "<response_a>\nbeta-" in prompt else "B"
        response = f"opaque judge rationale\nPAIRWISE_DECISION: {decision}"
        judge_text.add(response)
        return ModelCallResult(
            name=judge.name,
            model_id=judge.model,
            response=response,
            provider="local-test",
            latency_s=2.0,
        )

    report = asyncio.run(
        assay.run_assay(
            concurrency=4,
            timeout=47.0,
            candidate_runner=fake_candidates,
            judge_runner=fake_judge,
        )
    )
    serialized = json.dumps(report, sort_keys=True)

    assert report["responses_retained"] is False
    assert report["request_contract"]["fireworks_client_contract"] == (
        "fireworks-us-k3-priority-v1"
    )
    assert report["request_contract"]["fireworks_requested_service_tier"] == "priority"
    assert report["ordering"]["seed"] == assay.PREDECLARED_SEED
    assert report["decision"]["action"] == "eligible-for-3-of-3-canary"
    assert report["decision"]["fireworks_points"] == 40.0
    assert len(report["candidate_attempts"]) == 40
    assert len(report["judgments"]) == 40
    assert report["ordering"]["balanced"] is True
    assert all(text not in serialized for text in candidate_text | judge_text)
    assert all(case.question not in serialized for case in assay.CASES)

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    assert "response" not in set(keys(report))


def test_admission_decision_does_not_mutate_inputs():
    routes = _passing_route_summaries()
    judges = _passing_judge_summaries()
    ordering = assay.summarize_order_plan(assay.build_order_plan(129), 129)
    before = copy.deepcopy((routes, judges, ordering))

    assay.admission_decision(routes, judges, ordering)

    assert (routes, judges, ordering) == before
