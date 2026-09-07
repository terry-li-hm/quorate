"""Tests for council quorum and judge fallback behavior."""

import asyncio

from rich.console import Console

from quorate.config import ModelCallResult, ModelEntry
from quorate.modes import council


def test_k3_outcome_telemetry_recognises_incumbent_and_fireworks_routes():
    assert council._has_successful_k3(
        [ModelCallResult("K3", "k3", "answer", provider="kimi-code-api")]
    )
    assert council._has_successful_k3(
        [
            ModelCallResult(
                "Kimi-K3-US",
                "kimi-k3-us",
                "answer",
                provider="fireworks-us",
            )
        ]
    )
    assert not council._has_successful_k3(
        [
            ModelCallResult(
                "Kimi-K3-US",
                "kimi-k3-us",
                "[Error: route unavailable]",
                provider="none",
            )
        ]
    )


def test_council_uses_subscription_judge_fallback(monkeypatch):
    models = [ModelEntry(f"Model-{index}", f"vendor/model-{index}") for index in range(4)]

    async def fake_run_parallel(*_args, **_kwargs):
        return [
            ModelCallResult(entry.name, entry.model, f"Claim {index}", provider="test")
            for index, entry in enumerate(models)
        ]

    async def fake_query_judge(model, *_args, **_kwargs):
        if model == "google/gemini-3.5-flash":
            return "[Error: preferred judge unavailable]"
        assert model == "openai/gpt-5.6-sol"
        return "Fallback synthesis"

    monkeypatch.setattr(council, "run_parallel", fake_run_parallel)
    monkeypatch.setattr(council, "query_judge", fake_query_judge)
    monkeypatch.setattr(council.runlog, "append", lambda _record: None)

    result = asyncio.run(
        council.run_council(
            "question",
            models=models,
            rounds=0,
            no_critic=True,
            judge_model="google/gemini-3.5-flash",
            console=Console(quiet=True),
            json_output=True,
        )
    )

    assert isinstance(result, dict)
    assert result["quorum_achieved"] is True
    assert result["phases"]["judge"]["model"] == "openai/gpt-5.6-sol"
    assert result["phases"]["judge"]["preferred_model"] == "google/gemini-3.5-flash"


def test_council_uses_subscription_critic_fallback(monkeypatch):
    models = [ModelEntry(f"Model-{index}", f"vendor/model-{index}") for index in range(4)]

    async def fake_run_parallel(*_args, **_kwargs):
        return [
            ModelCallResult(entry.name, entry.model, f"Claim {index}", provider="test")
            for index, entry in enumerate(models)
        ]

    async def fake_query_judge(model, *_args, **_kwargs):
        if model == "anthropic/claude-fable-5-1":
            return "Primary synthesis"
        if model == "google/gemini-3.5-flash":
            return "[Error: preferred critic unavailable]"
        assert model == "anthropic/claude-opus-5"
        return "Fallback critique"

    monkeypatch.setattr(council, "run_parallel", fake_run_parallel)
    monkeypatch.setattr(council, "query_judge", fake_query_judge)
    monkeypatch.setattr(council.runlog, "append", lambda _record: None)

    result = asyncio.run(
        council.run_council(
            "question",
            models=models,
            rounds=0,
            console=Console(quiet=True),
            json_output=True,
        )
    )

    assert isinstance(result, dict)
    assert result["phases"]["critique"]["model"] == "anthropic/claude-opus-5"
    assert result["phases"]["critique"]["preferred_model"] == "google/gemini-3.5-flash"


def test_council_usage_includes_debate_calls(monkeypatch):
    models = [ModelEntry(f"Model-{index}", f"vendor/model-{index}") for index in range(4)]
    appended = []

    async def fake_run_parallel(*_args, **_kwargs):
        return [
            ModelCallResult(
                entry.name,
                entry.model.rsplit("/", 1)[-1],
                f"Claim {index}",
                provider="test",
                tokens_in=10,
                tokens_out=5,
            )
            for index, entry in enumerate(models)
        ]

    async def fake_model_runner(entry, *_args, **_kwargs):
        return ModelCallResult(
            entry.name,
            entry.model.rsplit("/", 1)[-1],
            "Debate response",
            provider="test",
            tokens_in=20,
            tokens_out=10,
        )

    async def fake_judge(*_args, **_kwargs):
        return "Synthesis"

    monkeypatch.setattr(council.runlog, "append", appended.append)

    asyncio.run(
        council.run_council(
            "question",
            models=models,
            rounds=1,
            no_critic=True,
            console=Console(quiet=True),
            json_output=True,
            parallel_runner=fake_run_parallel,
            model_runner=fake_model_runner,
            judge_runner=fake_judge,
        )
    )

    assert len(appended) == 1
    assert sum(row["calls"] for row in appended[0].model_usage) == 8
    assert sum(row["tokens_in"] for row in appended[0].model_usage) == 120
    assert sum(row["tokens_out"] for row in appended[0].model_usage) == 60


def test_council_logs_failed_blind_phase(monkeypatch):
    models = [ModelEntry(f"Model-{index}", f"vendor/model-{index}") for index in range(4)]
    appended = []

    async def fake_run_parallel(*_args, **_kwargs):
        return [
            ModelCallResult(
                entry.name,
                entry.model.rsplit("/", 1)[-1],
                "Claim" if index == 0 else "[Error: unavailable]",
                provider="test" if index == 0 else "none",
            )
            for index, entry in enumerate(models)
        ]

    monkeypatch.setattr(council.runlog, "append", appended.append)

    result = asyncio.run(
        council.run_council(
            "question",
            models=models,
            console=Console(quiet=True),
            json_output=True,
            parallel_runner=fake_run_parallel,
        )
    )

    assert isinstance(result, dict)
    assert result["quorum_achieved"] is False
    assert len(appended) == 1
    assert sum(row["reachable"] for row in appended[0].model_usage) == 1


def test_council_logs_member_calls_when_judge_fails(monkeypatch):
    models = [ModelEntry(f"Model-{index}", f"vendor/model-{index}") for index in range(4)]
    appended = []

    async def fake_run_parallel(*_args, **_kwargs):
        return [
            ModelCallResult(
                entry.name,
                entry.model.rsplit("/", 1)[-1],
                "Claim",
                provider="test",
            )
            for entry in models
        ]

    async def fake_judge(*_args, **_kwargs):
        return "[Error: judge unavailable]"

    monkeypatch.setattr(council.runlog, "append", appended.append)

    result = asyncio.run(
        council.run_council(
            "question",
            models=models,
            rounds=0,
            no_critic=True,
            judge_model="anthropic/claude-fable-5",
            judge_fallback_model="anthropic/claude-fable-5",
            console=Console(quiet=True),
            json_output=True,
            parallel_runner=fake_run_parallel,
            judge_runner=fake_judge,
        )
    )

    assert isinstance(result, dict)
    assert result["phases"]["judge"]["error"] == "[Error: judge unavailable]"
    assert len(appended) == 1
    assert sum(row["calls"] for row in appended[0].model_usage) == 4


def test_council_scores_blind_divergence_before_debate(monkeypatch):
    models = [ModelEntry(f"Model-{index}", f"vendor/model-{index}") for index in range(4)]
    appended = []
    judge_prompts = []

    async def fake_run_parallel(*_args, **_kwargs):
        return [
            ModelCallResult(entry.name, entry.model, f"Claim {index}", provider="test")
            for index, entry in enumerate(models)
        ]

    async def fake_judge(_model, messages, *_args, **_kwargs):
        judge_prompts.append(messages[1].content)
        if "measurement, not judgment" in messages[0].content:
            return (
                '{"positions": [{"label": "Ship it", "members": ["Model-0", "Model-1",'
                ' "Model-2"]}, {"label": "Wait", "members": ["Model-3"]}]}'
            )
        return "Synthesis"

    monkeypatch.setattr(council.runlog, "append", appended.append)

    result = asyncio.run(
        council.run_council(
            "question",
            models=models,
            rounds=0,
            no_critic=True,
            console=Console(quiet=True),
            json_output=True,
            parallel_runner=fake_run_parallel,
            judge_runner=fake_judge,
        )
    )

    assert isinstance(result, dict)
    assert result["divergence"]["index"] == 25.0
    assert result["divergence"]["distinct_positions"] == 2
    assert result["divergence"]["largest_bloc"] == 3
    # The score must be taken before the debate, or convergence is the protocol's.
    assert "--- DEBATE ---" not in judge_prompts[0]
    assert appended[0].divergence["index"] == 25.0


def test_council_survives_an_unusable_divergence_score(monkeypatch):
    models = [ModelEntry(f"Model-{index}", f"vendor/model-{index}") for index in range(4)]
    appended = []

    async def fake_run_parallel(*_args, **_kwargs):
        return [
            ModelCallResult(entry.name, entry.model, f"Claim {index}", provider="test")
            for index, entry in enumerate(models)
        ]

    async def fake_judge(_model, messages, *_args, **_kwargs):
        if "measurement, not judgment" in messages[0].content:
            return "I am afraid I cannot score that."
        return "Synthesis"

    monkeypatch.setattr(council.runlog, "append", appended.append)

    result = asyncio.run(
        council.run_council(
            "question",
            models=models,
            rounds=0,
            no_critic=True,
            console=Console(quiet=True),
            json_output=True,
            parallel_runner=fake_run_parallel,
            judge_runner=fake_judge,
        )
    )

    assert isinstance(result, dict)
    assert "divergence" not in result
    assert result["phases"]["judge"]["response"] == "Synthesis"
    assert appended[0].divergence is None


def test_council_skips_divergence_when_disabled(monkeypatch):
    models = [ModelEntry(f"Model-{index}", f"vendor/model-{index}") for index in range(4)]
    judge_calls = []

    async def fake_run_parallel(*_args, **_kwargs):
        return [
            ModelCallResult(entry.name, entry.model, f"Claim {index}", provider="test")
            for index, entry in enumerate(models)
        ]

    async def fake_judge(_model, messages, *_args, **_kwargs):
        judge_calls.append(messages[0].content)
        return "Synthesis"

    monkeypatch.setattr(council.runlog, "append", lambda _record: None)

    asyncio.run(
        council.run_council(
            "question",
            models=models,
            rounds=0,
            no_critic=True,
            score_divergence=False,
            console=Console(quiet=True),
            json_output=True,
            parallel_runner=fake_run_parallel,
            judge_runner=fake_judge,
        )
    )

    assert len(judge_calls) == 1
