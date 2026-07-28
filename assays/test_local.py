"""Tests for fail-closed local workspace grounding."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from rich.console import Console

from quorate import cli
from quorate.api import quorum_health
from quorate.config import Message, ModelCallResult
from quorate.local import (
    CODEX_ENTRY,
    GEMINI_ENTRY,
    LocalRunner,
    WorkspaceBoundaryError,
    build_workspace_snapshot,
    local_models,
)
from quorate.modes import brainstorm, council, quick


def test_snapshot_includes_text_and_reports_binary(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("public synthetic note")
    (tmp_path / "image.bin").write_bytes(b"\x00\x01")

    snapshot = build_workspace_snapshot(tmp_path)

    assert snapshot.file_count == 1
    assert snapshot.byte_count == len(b"public synthetic note")
    assert "--- FILE: notes.txt ---" in snapshot.text
    assert snapshot.omitted_binary_files == ("image.bin",)


@pytest.mark.parametrize("workspace_side", ["inside", "ancestor"])
def test_snapshot_refuses_protected_root_overlap(tmp_path: Path, workspace_side: str):
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "secret.txt").write_text("secret")
    workspace = protected if workspace_side == "inside" else tmp_path

    with pytest.raises(WorkspaceBoundaryError, match="overlaps configured protected root"):
        build_workspace_snapshot(workspace, protected_roots=(protected,))


def test_snapshot_refuses_symlink(tmp_path: Path):
    target = tmp_path / "target.txt"
    target.write_text("public")
    (tmp_path / "link.txt").symlink_to(target)

    with pytest.raises(WorkspaceBoundaryError, match="Symbolic links are not allowed"):
        build_workspace_snapshot(tmp_path)


@pytest.mark.parametrize("filename", [".env", ".env.local", "private.pem"])
def test_snapshot_refuses_sensitive_looking_file(tmp_path: Path, filename: str):
    (tmp_path / filename).write_text("TOKEN=synthetic")

    with pytest.raises(WorkspaceBoundaryError, match="Sensitive-looking file"):
        build_workspace_snapshot(tmp_path)


def test_snapshot_refuses_file_limit(tmp_path: Path, monkeypatch):
    (tmp_path / "one.txt").write_text("one")
    (tmp_path / "two.txt").write_text("two")
    monkeypatch.setattr("quorate.local.MAX_WORKSPACE_FILES", 1)

    with pytest.raises(WorkspaceBoundaryError, match="limit is 1"):
        build_workspace_snapshot(tmp_path)


def test_snapshot_refuses_per_file_limit(tmp_path: Path, monkeypatch):
    (tmp_path / "large.txt").write_text("four")
    monkeypatch.setattr("quorate.local.MAX_FILE_BYTES", 3)

    with pytest.raises(WorkspaceBoundaryError, match="exceeds 3 bytes"):
        build_workspace_snapshot(tmp_path)


def test_snapshot_refuses_aggregate_limit(tmp_path: Path, monkeypatch):
    (tmp_path / "one.txt").write_text("one")
    (tmp_path / "two.txt").write_text("two")
    monkeypatch.setattr("quorate.local.MAX_SNAPSHOT_BYTES", 5)

    with pytest.raises(WorkspaceBoundaryError, match="text exceeds 5 bytes"):
        build_workspace_snapshot(tmp_path)


def test_cli_requires_local_and_workspace_together(tmp_path: Path):
    with pytest.raises(SystemExit):
        cli._resolve_local_runner(True, None)
    with pytest.raises(SystemExit):
        cli._resolve_local_runner(False, str(tmp_path))


def test_local_runner_calls_only_fixed_subscription_clients(tmp_path: Path, monkeypatch):
    (tmp_path / "FACT.txt").write_text("ORCHID-731")
    runner = LocalRunner(build_workspace_snapshot(tmp_path))
    calls: list[tuple[str, str]] = []

    async def fake_codex(model, messages, _timeout, _effort):
        calls.append(("codex", model))
        assert "ORCHID-731" in messages[1].content
        return "codex answer", None

    async def fake_antigravity(model, messages, _timeout, _effort):
        calls.append(("gemini", model))
        assert "ORCHID-731" in messages[1].content
        return "gemini answer", None

    monkeypatch.setattr("quorate.local._codex_exec", fake_codex)
    monkeypatch.setattr("quorate.local._antigravity_prompt", fake_antigravity)

    results = asyncio.run(
        runner.run_parallel(local_models(), [Message.user("Return the token")], timeout=5)
    )

    assert calls == [
        ("codex", "openai/gpt-5.6-sol"),
        ("gemini", "google/gemini-3.5-flash"),
    ]
    assert [result.provider for result in results] == ["codex-exec", "antigravity-cli"]
    assert quorum_health(results) == (2, 2, True)
    results[1].response = "[Error: unavailable]"
    assert quorum_health(results) == (1, 2, False)


def test_quick_uses_injected_local_parallel_runner(monkeypatch):
    calls = []

    async def fake_parallel(models, messages, **_kwargs):
        calls.append((models, messages))
        return [
            ModelCallResult(CODEX_ENTRY.name, CODEX_ENTRY.model, "Codex answer"),
            ModelCallResult(GEMINI_ENTRY.name, GEMINI_ENTRY.model, "Gemini answer"),
        ]

    monkeypatch.setattr(quick.runlog, "append", lambda _record: None)
    result = asyncio.run(
        quick.run_quick(
            "question",
            models=local_models(),
            parallel_runner=fake_parallel,
            console=Console(quiet=True),
            json_output=True,
        )
    )

    assert calls
    assert result["quorum_achieved"] is True


def test_brainstorm_uses_injected_local_runners(monkeypatch):
    prompt_calls = []
    judge_calls = []

    async def fake_prompt_round(entries, prompts, **_kwargs):
        prompt_calls.append((entries, prompts))
        return [
            ModelCallResult(entry.name, entry.model, f"ideas from {entry.name}")
            for entry in entries
        ]

    async def fake_judge(model, _messages, **_kwargs):
        judge_calls.append(model)
        return "## Shortlist\nSix\n\n## Wildcard\nOne\n\n## Pattern map\nTwo"

    monkeypatch.setattr(brainstorm.runlog, "append", lambda _record: None)
    result = asyncio.run(
        brainstorm.run_brainstorm(
            "question",
            models=local_models(),
            prompt_round_runner=fake_prompt_round,
            judge_runner=fake_judge,
            curator_model=CODEX_ENTRY.model,
            curator_fallback_model=GEMINI_ENTRY.model,
            console=Console(quiet=True),
            json_output=True,
        )
    )

    assert len(prompt_calls) == 2
    assert judge_calls == [CODEX_ENTRY.model]
    assert result["quorum_achieved"] is True


def test_council_uses_injected_local_runners(monkeypatch):
    sequential_calls = []
    judge_calls = []

    async def fake_parallel(models, _messages, **_kwargs):
        return [
            ModelCallResult(entry.name, entry.model, f"blind from {entry.name}") for entry in models
        ]

    async def fake_model(entry, _messages, **_kwargs):
        sequential_calls.append(entry.model)
        return ModelCallResult(entry.name, entry.model, f"debate from {entry.name}")

    async def fake_judge(model, _messages, **_kwargs):
        judge_calls.append(model)
        return f"answer from {model}"

    monkeypatch.setattr(council.runlog, "append", lambda _record: None)
    monkeypatch.setattr(
        council.runlog,
        "prompt_outcome",
        lambda **_kwargs: (None, None, None),
    )
    result = asyncio.run(
        council.run_council(
            "question",
            models=local_models(),
            rounds=1,
            judge_model=CODEX_ENTRY.model,
            critic_model=GEMINI_ENTRY.model,
            judge_fallback_model=GEMINI_ENTRY.model,
            critic_fallback_model=CODEX_ENTRY.model,
            parallel_runner=fake_parallel,
            model_runner=fake_model,
            judge_runner=fake_judge,
            console=Console(quiet=True),
            json_output=True,
        )
    )

    assert sequential_calls == [CODEX_ENTRY.model, GEMINI_ENTRY.model]
    assert judge_calls == [CODEX_ENTRY.model, GEMINI_ENTRY.model]
    assert result["quorum_achieved"] is True
