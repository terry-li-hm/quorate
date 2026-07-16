"""Tests for cli.py _resolve_question / _resolve_context OSError handling."""

from pathlib import Path

import pytest

from quorate.cli import _emit_result, _resolve_context, _resolve_question


def test_resolve_question_long_string_returns_unchanged():
    long_input = "x" * 300
    result = _resolve_question(long_input)
    assert result == long_input


def test_resolve_context_long_string_returns_inline():
    long_input = "y" * 300
    result = _resolve_context((long_input,))
    assert result is not None
    assert long_input in result


def test_resolve_question_reads_real_file(tmp_path: Path):
    f = tmp_path / "q.txt"
    f.write_text("  hello from file  \n")
    result = _resolve_question(str(f))
    assert result == "hello from file"


def test_resolve_question_refuses_configured_protected_root(tmp_path: Path, monkeypatch):
    protected = tmp_path / "protected"
    protected.mkdir()
    question = protected / "question.md"
    question.write_text("confidential")
    monkeypatch.setenv("QUORATE_PROTECTED_ROOTS", str(protected))

    with pytest.raises(SystemExit):
        _resolve_question(str(question))


def test_resolve_context_refuses_symlink_into_protected_root(tmp_path: Path, monkeypatch):
    protected = tmp_path / "protected"
    protected.mkdir()
    context = protected / "context.md"
    context.write_text("confidential")
    link = tmp_path / "apparently-safe.md"
    link.symlink_to(context)
    monkeypatch.setenv("QUORATE_PROTECTED_ROOTS", str(protected))

    with pytest.raises(SystemExit):
        _resolve_context((str(link),))


def test_resolve_context_allows_file_outside_protected_root(tmp_path: Path, monkeypatch):
    protected = tmp_path / "protected"
    protected.mkdir()
    context = tmp_path / "public-context.md"
    context.write_text("public material")
    monkeypatch.setenv("QUORATE_PROTECTED_ROOTS", str(protected))

    assert _resolve_context((str(context),)) == "public material"


def test_emit_result_marks_missing_quorum_as_error(capsys):
    result = {
        "responses": [],
        "success_count": 1,
        "failed_count": 6,
        "quorum_target": 4,
        "quorum_achieved": False,
    }

    with pytest.raises(SystemExit):
        _emit_result("quorate quick", result, json_output=True)

    envelope = capsys.readouterr().out
    assert '"ok": false' in envelope
    assert '"result"' in envelope
    assert '"quorum_achieved": false' in envelope
