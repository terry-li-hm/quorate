"""Tests for cli.py _resolve_question / _resolve_context OSError handling."""

from pathlib import Path

from quorate.cli import _resolve_context, _resolve_question


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
