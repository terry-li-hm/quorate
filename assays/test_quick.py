"""Tests for quick-mode health reporting."""

import asyncio

from rich.console import Console

from quorate.config import ModelCallResult, ModelEntry
from quorate.modes import quick


def test_quick_json_reports_missing_quorum(monkeypatch):
    async def fake_run_parallel(*_args, **_kwargs):
        return [
            ModelCallResult("One", "one", "answer", provider="subscription"),
            ModelCallResult(
                "Two",
                "two",
                "[Error: All providers failed]",
                provider="none",
                diagnostics=("openrouter:http_404",),
            ),
            ModelCallResult("Three", "three", "[Error: failed]", provider="none"),
            ModelCallResult("Four", "four", "[Error: failed]", provider="none"),
        ]

    monkeypatch.setattr(quick, "run_parallel", fake_run_parallel)
    monkeypatch.setattr(quick.runlog, "append", lambda _record: None)
    models = [ModelEntry(str(index), str(index)) for index in range(4)]

    result = asyncio.run(
        quick.run_quick(
            "question",
            models=models,
            console=Console(quiet=True),
            json_output=True,
        )
    )

    assert isinstance(result, dict)
    assert result["success_count"] == 1
    assert result["quorum_target"] == 3
    assert result["quorum_achieved"] is False
    assert result["responses"][1]["diagnostics"] == ["openrouter:http_404"]
