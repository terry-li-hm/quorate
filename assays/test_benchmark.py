"""Tests for synthetic roster benchmarking."""

from __future__ import annotations

import asyncio
import json

from quorate import benchmark
from quorate.config import ModelCallResult, ModelEntry


def _result(entry: ModelEntry, response: str, latency: float = 1.0) -> ModelCallResult:
    return ModelCallResult(
        name=entry.name,
        model_id=entry.model.rsplit("/", 1)[-1],
        response=response,
        provider="test-route",
        latency_s=latency,
    )


def test_canary_validators_are_strict():
    exact, structured, reasoning = benchmark.CANARIES
    assert exact.validator("QUORATE_OK") is True
    assert exact.validator("QUORATE_OK because...") is False
    assert structured.validator('{"status":"ok","items":[1,2,3]}') is True
    assert structured.validator("```json\n{}\n```") is False
    assert reasoning.validator("42") is True
    assert reasoning.validator("The answer is 42") is False


def test_one_failed_seat_is_healthy(monkeypatch, tmp_path):
    models = [ModelEntry(f"M{i}", f"provider/m{i}") for i in range(7)]
    responses = {
        "exact-token": "QUORATE_OK",
        "json-contract": '{"status":"ok","items":[1,2,3]}',
        "simple-reasoning": "42",
    }
    call = 0

    async def fake_parallel(entries, messages, **_kwargs):
        nonlocal call
        canary = benchmark.CANARIES[call]
        call += 1
        rows = [_result(entry, responses[canary.id]) for entry in entries]
        rows[-1].response = "[Error: route unavailable]"
        rows[-1].provider = "none"
        rows[-1].diagnostics = ("test-route:timeout",)
        return rows

    monkeypatch.setattr(benchmark, "run_parallel", fake_parallel)
    report = asyncio.run(benchmark.run_benchmark(models=models, snapshot_dir=tmp_path))

    assert report["status"] == "healthy"
    assert report["quorum_achieved"] is True
    assert report["weak_seats"] == ["M6"]
    assert report["models"][-1]["diagnostics"] == ["test-route:timeout"]
    assert (tmp_path / f"{report['ts'][:10]}.json").exists()


def test_missing_canary_quorum_is_unhealthy(monkeypatch):
    models = [ModelEntry(f"M{i}", f"provider/m{i}") for i in range(7)]

    async def fake_parallel(entries, messages, **_kwargs):
        return [
            _result(entry, "QUORATE_OK" if index < 3 else "wrong")
            for index, entry in enumerate(entries)
        ]

    monkeypatch.setattr(benchmark, "run_parallel", fake_parallel)
    report = asyncio.run(benchmark.run_benchmark(models=models, save=False))

    assert report["status"] == "unhealthy"
    assert report["quorum_achieved"] is False
    assert all(canary["quorum_achieved"] is False for canary in report["canaries"])


def test_snapshot_is_idempotent_for_utc_date(tmp_path):
    report = {"ts": "2026-07-16T00:00:00+00:00", "status": "healthy"}
    first = benchmark.save_report(report, tmp_path)
    report["status"] = "degraded"
    second = benchmark.save_report(report, tmp_path)

    assert first == second
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert json.loads(second.read_text())["status"] == "degraded"
