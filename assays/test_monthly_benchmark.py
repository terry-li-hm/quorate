"""Tests for the monthly Fireworks exact-route gate."""

import json
import subprocess
import sys
from pathlib import Path

FIREWORKS_US_MODEL = "accounts/fireworks/routers/kimi-k3-us"
CHECKER = Path(__file__).parents[1] / "scripts" / "check_fireworks_benchmark.py"


def _payload(row=None):
    models = [] if row is None else [row]
    return {"result": {"models": models}}


def _row(**overrides):
    row = {
        "model_id": FIREWORKS_US_MODEL,
        "canaries": 3,
        "reachable": 3,
        "passed": 3,
        "providers": {"fireworks-us": 3},
        "request_contract": {
            "client_contract": "fireworks-us-k3-priority-v1",
            "requested_service_tier": "priority",
        },
    }
    row.update(overrides)
    return row


def _run(tmp_path, payload, *, required=False):
    report = tmp_path / "benchmark.json"
    report.write_text(json.dumps(payload))
    arguments = [sys.executable, str(CHECKER)]
    if required:
        arguments.append("--required")
    arguments.append(str(report))
    return subprocess.run(
        arguments,
        check=False,
    ).returncode


def test_fireworks_gate_ignores_absent_candidate(tmp_path):
    assert _run(tmp_path, _payload()) == 2


def test_fireworks_gate_rejects_absent_candidate_when_selected(tmp_path):
    assert _run(tmp_path, _payload(), required=True) == 1


def test_fireworks_gate_accepts_exact_three_of_three_route(tmp_path):
    assert _run(tmp_path, _payload(_row()), required=True) == 0


def test_fireworks_gate_rejects_wrong_provider_or_diagnostics(tmp_path):
    assert _run(tmp_path, _payload(_row(providers={"none": 3}))) == 1
    assert _run(tmp_path, _payload(_row(diagnostics=["fireworks-us:http_503"]))) == 1


def test_fireworks_gate_rejects_partial_canary_pass(tmp_path):
    assert _run(tmp_path, _payload(_row(passed=2))) == 1


def test_fireworks_gate_rejects_unattested_or_standard_tier(tmp_path):
    assert _run(tmp_path, _payload(_row(request_contract=None))) == 1
    assert (
        _run(
            tmp_path,
            _payload(
                _row(
                    request_contract={
                        "client_contract": "fireworks-us-k3-priority-v1",
                        "requested_service_tier": "standard",
                    }
                )
            ),
        )
        == 1
    )
