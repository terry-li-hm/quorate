"""Tests for divergent brainstorm generation, cross-pollination, and curation."""

import asyncio

from rich.console import Console

from quorate.config import ModelCallResult, ModelEntry
from quorate.modes import brainstorm


def _models(count: int = 4) -> list[ModelEntry]:
    return [ModelEntry(f"Model-{index}", f"vendor/model-{index}") for index in range(count)]


def test_brainstorm_runs_two_rounds_then_curates(monkeypatch):
    models = _models()

    async def fake_query_model(_client, _keys, entry, messages, **_kwargs):
        phase = "hybrid" if "cross-pollination" in messages[0].content else "seed"
        return ModelCallResult(
            name=entry.name,
            model_id=entry.model,
            response=f"{phase} ideas from {entry.name}",
            provider="test",
        )

    async def fake_query_judge(model, *_args, **_kwargs):
        assert model == "anthropic/claude-fable-5"
        return "## Shortlist\nCurated ideas\n\n## Wildcard\nOne bet\n\n## Pattern map\nA pattern"

    monkeypatch.setattr(brainstorm, "query_model", fake_query_model)
    monkeypatch.setattr(brainstorm, "query_judge", fake_query_judge)
    monkeypatch.setattr(brainstorm.runlog, "append", lambda _record: None)

    result = asyncio.run(
        brainstorm.run_brainstorm(
            "Invent a better archive",
            models=models,
            console=Console(quiet=True),
            json_output=True,
        )
    )

    assert isinstance(result, dict)
    assert result["quorum_achieved"] is True
    assert len(result["phases"]["generation"]) == 4
    assert len(result["phases"]["cross_pollination"]) == 4
    assert result["phases"]["curation"]["model"] == "anthropic/claude-fable-5"
    assert result["phases"]["generation"][0]["lens"] == "First principles"
    assert result["phases"]["cross_pollination"][0]["source_model"] == "Model-1"


def test_brainstorm_stops_before_cross_pollination_without_quorum(monkeypatch):
    models = _models()

    async def fake_query_model(_client, _keys, entry, _messages, **_kwargs):
        response = "one seed" if entry.name == "Model-0" else "[Error: unavailable]"
        return ModelCallResult(entry.name, entry.model, response, provider="test")

    monkeypatch.setattr(brainstorm, "query_model", fake_query_model)

    result = asyncio.run(
        brainstorm.run_brainstorm(
            "Invent a better archive",
            models=models,
            console=Console(quiet=True),
            json_output=True,
        )
    )

    assert isinstance(result, dict)
    assert result["quorum_achieved"] is False
    assert set(result["phases"]) == {"generation"}


def test_brainstorm_falls_back_when_curator_misses_contract(monkeypatch):
    models = _models()

    async def fake_query_model(_client, _keys, entry, _messages, **_kwargs):
        return ModelCallResult(entry.name, entry.model, "ideas", provider="test")

    async def fake_query_judge(model, *_args, **_kwargs):
        if model == "anthropic/claude-fable-5":
            return "A caveat without the requested shortlist"
        assert model == "openai/gpt-5.6-sol"
        return "## Shortlist\nSix ideas\n\n## Wildcard\nOne bet\n\n## Pattern map\nA pattern"

    monkeypatch.setattr(brainstorm, "query_model", fake_query_model)
    monkeypatch.setattr(brainstorm, "query_judge", fake_query_judge)
    monkeypatch.setattr(brainstorm.runlog, "append", lambda _record: None)

    result = asyncio.run(
        brainstorm.run_brainstorm(
            "Invent a better archive",
            models=models,
            console=Console(quiet=True),
            json_output=True,
        )
    )

    assert isinstance(result, dict)
    assert result["phases"]["curation"]["model"] == "openai/gpt-5.6-sol"
    assert result["phases"]["curation"]["preferred_model"] == "anthropic/claude-fable-5"
