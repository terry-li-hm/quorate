from __future__ import annotations

from quorate.config import ModelCallResult
from quorate.judge_assay import CASES, FABLE, GPT, attempt_row, parse_decision, routing_decision


def test_suite_is_balanced_and_unique():
    assert len(CASES) == 20
    assert len({case.id for case in CASES}) == 20
    assert sum(case.expected == "A" for case in CASES) == 10
    assert sum(case.expected == "B" for case in CASES) == 10


def test_decision_parser_requires_exactly_one_marker():
    assert parse_decision("Reasoning\nFINAL_DECISION: A") == "A"
    assert parse_decision("final_decision: b") == "B"
    assert parse_decision("A") is None
    assert parse_decision("FINAL_DECISION: A\nFINAL_DECISION: B") is None


def test_attempt_row_does_not_retain_response():
    result = ModelCallResult(
        name=FABLE.name,
        model_id="claude-fable-5",
        response="Synthetic reasoning\nFINAL_DECISION: A",
        provider="test",
        latency_s=1.25,
    )
    row = attempt_row(CASES[0], result)
    assert row["correct"] is True
    assert row["response_chars"] > 0
    assert "response" not in row


def test_gpt_requires_predeclared_quality_margin_to_replace_fable():
    tied = [
        {
            "model": FABLE.name,
            "reachable": 20,
            "correct": 20,
            "contract_completed": 20,
        },
        {"model": GPT.name, "reachable": 20, "correct": 20, "contract_completed": 20},
    ]
    decisive = [
        {
            "model": FABLE.name,
            "reachable": 20,
            "correct": 18,
            "contract_completed": 20,
        },
        {"model": GPT.name, "reachable": 20, "correct": 20, "contract_completed": 20},
    ]
    incomplete = [
        {
            "model": FABLE.name,
            "reachable": 20,
            "correct": 17,
            "contract_completed": 20,
        },
        {"model": GPT.name, "reachable": 20, "correct": 19, "contract_completed": 18},
    ]
    route_outage = [
        {"model": FABLE.name, "reachable": 0, "correct": 0, "contract_completed": 0},
        {"model": GPT.name, "reachable": 20, "correct": 20, "contract_completed": 20},
    ]
    assert routing_decision(tied)["action"] == "retain-fable"
    assert routing_decision(decisive)["action"] == "promote-gpt"
    assert routing_decision(incomplete)["action"] == "retain-fable"
    assert routing_decision(route_outage)["action"] == "invalid-run"
