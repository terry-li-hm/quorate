"""Tests for quorate.seventh_seat_assay — panel-uniqueness harness, fake runners only."""

import asyncio
import json
import os
import re
import sys

import pytest

from quorate import seventh_seat_assay
from quorate.config import ModelCallResult
from quorate.council_seat_assay import CASES, FABLE, GPT
from quorate.seventh_seat_assay import (
    DEFAULT_CANDIDATE,
    PREFLIGHT_PROMPT,
    admission_decision,
    assay_judges,
    incumbent_seats,
    main,
    panel_order,
    parse_scores,
    run_assay,
)

UNJUDGED_CASE = CASES[0]
SEAT_TOKEN = "seat-answer-token"
JUDGE_TOKEN = "judge-reason-token"


def test_reuses_twenty_council_seat_cases():
    from quorate.seventh_seat_assay import CASES as suite_cases

    assert suite_cases is CASES
    assert len(suite_cases) == 20
    assert len({case.id for case in suite_cases}) == 20


def test_parse_scores_requires_one_complete_marker():
    valid = "Reason.\nSCORES: A=2 B=1 C=0 D=1"
    assert parse_scores(valid, "ABCD") == {"A": 2, "B": 1, "C": 0, "D": 1}
    assert parse_scores(f"{valid}\nSCORES: A=1 B=1 C=1 D=1", "ABCD") is None
    assert parse_scores("SCORES: A=2 B=1 C=0", "ABCD") is None
    assert parse_scores("SCORES: A=2 B=1 C=0 D=1 E=2", "ABCD") is None
    assert parse_scores("SCORES: A=2 A=1 B=1 C=0 D=1", "ABCD") is None
    assert parse_scores("SCORES: A=3 B=1 C=0 D=1", "ABCD") is None
    assert parse_scores("SCORES: A=2,B=1,C=0,D=1", "ABCD") == {"A": 2, "B": 1, "C": 0, "D": 1}


def test_panel_order_is_seeded_and_round_trips():
    names = ["GPT-6.1-Sol", "Claude-Opus-5", "Grok-4.5", "Kimi-K3", "GLM-5.2", "Gemini-4-Argon"]
    case = CASES[0]
    first = panel_order(case, FABLE.name, list(names))
    assert first == panel_order(case, FABLE.name, list(names))
    second = panel_order(case, GPT.name, list(names))
    assert sorted(first) == sorted(second) == sorted(names)
    assert len(set(first)) == len(names)


def test_incumbents_exclude_configured_seventh_seat(monkeypatch):
    monkeypatch.setenv("CONSILIUM_MODEL_M7", "google/gemini-4-argon")
    seats = incumbent_seats()
    assert len(seats) == 6
    assert all(seat.model != "google/gemini-4-argon" for seat in seats)
    assert os.environ["CONSILIUM_MODEL_M7"] == "google/gemini-4-argon"


def _seat_summary(
    model,
    *,
    role="incumbent",
    attempts=20,
    reachable=20,
    compliant=20,
    truncated=0,
    median_latency=20.0,
):
    return {
        "model": model,
        "role": role,
        "attempts": attempts,
        "reachable": reachable,
        "output_compliant": compliant,
        "truncated": truncated,
        "mean_latency_s": median_latency,
        "median_latency_s": median_latency,
    }


def _judge_summary(name, *, contracts=20):
    return {"judge": name, "cases": 20, "reachable": 20, "contract_completed": contracts}


INCUMBENT_NAMES = [
    "GPT-6.1-Sol",
    "Claude-Opus-5",
    "Grok-4.5",
    "Kimi-K3",
    "GLM-5.2",
    "DeepSeek-V4-Pro",
]


def _panel_inputs(**candidate_overrides):
    candidate = _seat_summary("Gemini-4-Argon", role="candidate", **candidate_overrides)
    seats = [_seat_summary(name) for name in INCUMBENT_NAMES]
    scores = {
        "Gemini-4-Argon": 1.5,
        "GPT-6.1-Sol": 1.0,
        "Claude-Opus-5": 1.2,
        "Grok-4.5": 1.4,
        "Kimi-K3": 1.6,
        "GLM-5.2": 1.8,
        "DeepSeek-V4-Pro": 2.0,
    }
    judges = [_judge_summary(FABLE.name), _judge_summary(GPT.name)]
    return candidate, seats, scores, judges


def test_admission_requires_median_and_every_floor():
    candidate, seats, scores, judges = _panel_inputs()
    decision = admission_decision([*seats, candidate], judges, scores, "Gemini-4-Argon")
    assert decision["action"] == "admit-trial"
    assert decision["candidate_mean"] == 1.5
    assert decision["incumbent_median"] == 1.5
    assert all(decision["floors"].values())

    below = dict(scores, **{"Gemini-4-Argon": 1.4})
    below_decision = admission_decision([*seats, candidate], judges, below, "Gemini-4-Argon")
    assert below_decision["action"] == "retain-six-seats"

    for overrides, floor in (
        ({"reachable": 18}, "reachable"),
        ({"compliant": 18}, "output_compliant"),
        ({"truncated": 2}, "not_truncated"),
        ({"median_latency": 20.5}, "median_latency_s"),
    ):
        candidate, seats, scores, judges = _panel_inputs(**overrides)
        decision = admission_decision([*seats, candidate], judges, scores, "Gemini-4-Argon")
        assert decision["action"] == "retain-six-seats"
        assert decision["floors"][floor] is False
        assert all(value for key, value in decision["floors"].items() if key != floor)


def test_admission_marks_invalid_runs():
    candidate, seats, scores, judges = _panel_inputs()
    weak = [
        dict(seat, reachable=17) if seat["model"] in {"Kimi-K3", "GLM-5.2"} else seat
        for seat in seats
    ]
    weak_decision = admission_decision([*weak, candidate], judges, scores, "Gemini-4-Argon")
    assert weak_decision["action"] == "invalid-run"

    weak_judges = [_judge_summary(FABLE.name), _judge_summary(GPT.name, contracts=17)]
    judge_decision = admission_decision([*seats, candidate], weak_judges, scores, "Gemini-4-Argon")
    assert judge_decision["action"] == "invalid-run"


def _seat_result(entry, question):
    if entry.model == DEFAULT_CANDIDATE and question == UNJUDGED_CASE.question:
        return ModelCallResult(entry.name, entry.model, "[Error: unreachable]", "none")
    body = f"{SEAT_TOKEN} {entry.name} " + "filler " * 60
    return ModelCallResult(entry.name, entry.model, body.strip(), "fake", 5.0)


def _fake_seat_runner(models, messages, max_tokens, timeout, effort):
    if messages[-1].content == PREFLIGHT_PROMPT:
        assert max_tokens == 256 and effort is None
        return [ModelCallResult(e.name, e.model, "READY", "fake", 0.1) for e in models]
    assert max_tokens == 1024 and effort is None
    question = messages[-1].content
    return [_seat_result(entry, question) for entry in models]


def _fake_judge_runner(models, messages, max_tokens, timeout, effort):
    if messages[-1].content == PREFLIGHT_PROMPT:
        assert max_tokens == 256 and effort is None
        return [ModelCallResult(e.name, e.model, "READY", "fake", 0.1) for e in models]
    assert max_tokens == 2048 and effort.value == "high"
    prompt = messages[-1].content
    labels = re.findall(r"<response_([A-G])>", prompt)
    assert labels, "judge prompt must contain labelled response blocks"
    pairs = []
    for label in labels:
        block = re.search(
            rf"<response_{label}>\n(.*?)\n</response_{label}>", prompt, re.DOTALL
        ).group(1)
        score = 2 if SEAT_TOKEN in block and "Gemini-4-Argon" in block else 1
        pairs.append(f"{label}={score}")
    response = f"{JUDGE_TOKEN} Distinct contribution noted.\nSCORES: {' '.join(pairs)}"
    judge = models[0]
    return [ModelCallResult(judge.name, judge.model, response, "fake", 7.0)]


def test_run_assay_with_fake_runners_scores_and_retains_no_text():
    report = asyncio.run(
        run_assay(
            seat_runner=_fake_seat_runner,
            judge_runner=_fake_judge_runner,
        )
    )

    assert report["candidate"]["model"] == DEFAULT_CANDIDATE
    assert report["cases"] == 20
    assert report["responses_retained"] is False
    assert report["judges"] == [
        {"name": "Claude-Fable-5-2", "model": "anthropic/claude-fable-5-2"},
        {"name": "GPT-6.1-Sol", "model": "openai/gpt-6.1-sol"},
    ]
    assert len(report["preflight"]) == 3
    assert all(row["reachable"] for row in report["preflight"])
    assert report["decision"]["action"] == "admit-trial"
    assert report["decision"]["candidate_mean"] == 2.0
    assert report["decision"]["incumbent_median"] == 1.0
    assert all(report["decision"]["floors"].values())

    judged_cases = {row["case"] for row in report["judgments"]}
    assert len(judged_cases) == 19
    assert UNJUDGED_CASE.id not in judged_cases
    assert len(report["judgments"]) == 38

    incumbent_means = [
        report["mean_scores"][name] for name in report["mean_scores"] if name != "Gemini-4-Argon"
    ]
    assert incumbent_means == [1.0] * 6

    dumped = json.dumps(report)
    assert SEAT_TOKEN not in dumped
    assert JUDGE_TOKEN not in dumped
    assert "filler" not in dumped
    assert "Distinct contribution noted" not in dumped


def test_main_writes_report(tmp_path, monkeypatch, capsys):
    payload = {
        "schema_version": 1,
        "decision": {"action": "retain-six-seats", "reason": "fake"},
    }

    async def fake_run_assay(*_args, **_kwargs):
        return payload

    monkeypatch.setattr(seventh_seat_assay, "run_assay", fake_run_assay)
    output = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(
        sys, "argv", ["seventh-seat-assay", "--output", str(output), "--concurrency", "2"]
    )

    assert main() == 0
    written = json.loads(output.read_text())
    assert written == payload
    printed = capsys.readouterr().out
    assert str(output) in printed
    assert "retain-six-seats" in printed


def test_default_judges_follow_council_judge_and_fallback(monkeypatch):
    monkeypatch.setenv("CONSILIUM_MODEL_JUDGE", "google/gemini-3.5-flash")
    monkeypatch.setenv("CONSILIUM_MODEL_JUDGE_FALLBACK", "anthropic/claude-opus-5")
    configured = assay_judges()
    assert [(judge.name, judge.model) for judge in configured] == [
        ("Gemini-3.5-Flash", "google/gemini-3.5-flash"),
        ("Claude-Opus-5", "anthropic/claude-opus-5"),
    ]

    monkeypatch.delenv("CONSILIUM_MODEL_JUDGE")
    monkeypatch.delenv("CONSILIUM_MODEL_JUDGE_FALLBACK")
    default = assay_judges()
    assert [(judge.name, judge.model) for judge in default] == [
        ("Claude-Fable-5-2", "anthropic/claude-fable-5-2"),
        ("GPT-6.1-Sol", "openai/gpt-6.1-sol"),
    ]


def test_assay_judges_require_two_distinct_models():
    pair = assay_judges(("google/gemini-4-argon", "anthropic/claude-opus-5"))
    assert [judge.model for judge in pair] == [
        "google/gemini-4-argon",
        "anthropic/claude-opus-5",
    ]
    for models in ((DEFAULT_CANDIDATE,), (DEFAULT_CANDIDATE, DEFAULT_CANDIDATE)):
        with pytest.raises(ValueError):
            assay_judges(models)


def _counting_runner(calls, kind, *, fail_preflight_for=None):
    def runner(models, messages, max_tokens, timeout, effort):
        entry = models[0]
        calls[kind].append((entry.model, messages[-1].content))
        if messages[-1].content == PREFLIGHT_PROMPT and entry.model == fail_preflight_for:
            return [ModelCallResult(entry.name, entry.model, "[Error: closed route]", "none")]
        return [ModelCallResult(entry.name, entry.model, "READY", "fake", 0.1)]

    return runner


def test_preflight_failure_returns_invalid_run_without_seat_queries():
    calls = {"seat": [], "judge": []}
    seat_runner = _counting_runner(calls, "seat", fail_preflight_for=DEFAULT_CANDIDATE)
    judge_runner = _counting_runner(calls, "judge")

    report = asyncio.run(run_assay(seat_runner=seat_runner, judge_runner=judge_runner))

    assert report["decision"]["action"] == "invalid-run"
    assert report["decision"]["reason"] == (
        "Preflight failed for: Gemini-4-Argon. No seat or judge queries were made."
    )
    assert report["decision"]["candidate_mean"] is None
    assert report["decision"]["incumbent_median"] is None
    assert report["decision"]["floors"] == {}
    assert report["seat_attempts"] == []
    assert report["seat_summaries"] == []
    assert report["judgments"] == []
    assert report["judge_summaries"] == []
    assert report["mean_scores"] == {}
    assert report["responses_retained"] is False
    assert [c for c in calls["seat"] if c[1] != PREFLIGHT_PROMPT] == []
    assert len(calls["judge"]) == 2
    assert all(c[1] == PREFLIGHT_PROMPT for c in calls["judge"])

    calls = {"seat": [], "judge": []}
    seat_runner = _counting_runner(calls, "seat")
    judge_runner = _counting_runner(calls, "judge", fail_preflight_for="anthropic/claude-fable-5-2")

    report = asyncio.run(run_assay(seat_runner=seat_runner, judge_runner=judge_runner))

    assert report["decision"]["reason"] == (
        "Preflight failed for: Claude-Fable-5-2. No seat or judge queries were made."
    )
    assert len(calls["seat"]) == 1
    assert all(c[1] == PREFLIGHT_PROMPT for c in calls["seat"])
    assert [c for c in calls["judge"] if c[1] != PREFLIGHT_PROMPT] == []


def test_candidate_floor_failure_retains_before_judge_validity():
    candidate, seats, scores, _judges = _panel_inputs(reachable=17, compliant=19)
    capped = [_judge_summary(FABLE.name, contracts=17), _judge_summary(GPT.name, contracts=17)]
    decision = admission_decision([*seats, candidate], capped, scores, "Gemini-4-Argon")
    assert decision["action"] == "retain-six-seats"
    assert "reachable" in decision["reason"]


def test_main_accepts_two_judges_and_rejects_one(tmp_path, monkeypatch, capsys):
    captured = {}

    async def fake_run_assay(*_args, **kwargs):
        captured.update(kwargs)
        return {"decision": {"action": "admit-trial", "reason": "fake"}}

    monkeypatch.setattr(seventh_seat_assay, "run_assay", fake_run_assay)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seventh-seat-assay",
            "--output",
            str(output),
            "--judge",
            "anthropic/claude-opus-5",
            "--judge",
            "google/gemini-4-argon",
        ],
    )
    assert main() == 0
    assert captured["judges"] == ("anthropic/claude-opus-5", "google/gemini-4-argon")
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        ["seventh-seat-assay", "--output", str(output), "--judge", "anthropic/claude-opus-5"],
    )
    with pytest.raises(SystemExit):
        main()
