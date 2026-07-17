from quorate.council_seat_assay import (
    CASES,
    FABLE,
    GPT,
    K2,
    K3_NAME,
    output_compliant,
    parse_kimi_stream,
    parse_pairwise,
    routing_decision,
)


def test_suite_has_twenty_distinct_cases():
    assert len(CASES) == 20
    assert len({case.id for case in CASES}) == 20


def test_output_contract():
    assert output_compliant("word " * 60)
    assert output_compliant("word " * 220)
    assert not output_compliant("word " * 59)
    assert not output_compliant("word " * 221)


def test_parse_pairwise_requires_one_marker():
    assert parse_pairwise("Reason.\nPAIRWISE_DECISION: A") == "A"
    assert parse_pairwise("PAIRWISE_DECISION: tie") == "TIE"
    assert parse_pairwise("A") is None
    assert parse_pairwise("PAIRWISE_DECISION: A\nPAIRWISE_DECISION: B") is None


def test_parse_kimi_stream_retains_only_assistant_content():
    payload = "\n".join(
        (
            '{"role":"assistant","content":"answer"}',
            '{"role":"meta","content":"resume hint"}',
        )
    )
    assert parse_kimi_stream(payload) == "answer"


def _candidate(model: str, *, latency: float = 20, reachable: int = 20, compliant: int = 20):
    return {
        "model": model,
        "reachable": reachable,
        "output_compliant": compliant,
        "within_timeout": reachable,
        "median_latency_s": latency,
    }


def _judge(name: str, points: float, *, reachable: int = 20, contracts: int = 20):
    return {
        "judge": name,
        "reachable": reachable,
        "contract_completed": contracts,
        "k3_points": points,
    }


def test_promotion_requires_quality_and_all_floors():
    candidates = [_candidate(K2.name), _candidate(K3_NAME, latency=40)]
    judges = [_judge(FABLE.name, 12), _judge(GPT.name, 12)]
    assert routing_decision(candidates, judges)["action"] == "replace-k2.6-with-k3"

    candidates[1] = _candidate(K3_NAME, latency=41)
    assert routing_decision(candidates, judges)["action"] == "retain-k2.6"

    candidates[1] = _candidate(K3_NAME, latency=40, compliant=18)
    assert routing_decision(candidates, judges)["action"] == "retain-k2.6"


def test_invalid_route_does_not_count_as_model_loss():
    candidates = [_candidate(K2.name, reachable=17), _candidate(K3_NAME)]
    judges = [_judge(FABLE.name, 15), _judge(GPT.name, 15)]
    assert routing_decision(candidates, judges)["action"] == "invalid-run"

    candidates = [_candidate(K2.name), _candidate(K3_NAME)]
    judges[0] = _judge(FABLE.name, 15, contracts=17)
    assert routing_decision(candidates, judges)["action"] == "invalid-run"
