"""Tests for council quorum and judge fallback behavior."""

import asyncio

from rich.console import Console

from quorate.config import ModelCallResult, ModelEntry
from quorate.modes import council


def test_council_uses_subscription_judge_fallback(monkeypatch):
    models = [ModelEntry(f"Model-{index}", f"vendor/model-{index}") for index in range(4)]

    async def fake_run_parallel(*_args, **_kwargs):
        return [
            ModelCallResult(entry.name, entry.model, f"Claim {index}", provider="test")
            for index, entry in enumerate(models)
        ]

    async def fake_query_judge(model, *_args, **_kwargs):
        if model == "google/gemini-3.1-pro-preview":
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
            judge_model="google/gemini-3.1-pro-preview",
            console=Console(quiet=True),
            json_output=True,
        )
    )

    assert isinstance(result, dict)
    assert result["quorum_achieved"] is True
    assert result["phases"]["judge"]["model"] == "openai/gpt-5.6-sol"
    assert result["phases"]["judge"]["preferred_model"] == "google/gemini-3.1-pro-preview"
