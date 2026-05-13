"""Tests for quorate.cli persona resolution — file load, prefix wrap, context merge."""

import pytest

from quorate.cli import (
    PERSONA_PREFIX,
    _resolve_persona,
    _merge_persona_context,
)


class TestResolvePersona:
    def test_none_returns_none(self):
        assert _resolve_persona(None) is None

    def test_empty_returns_none(self):
        assert _resolve_persona("") is None

    def test_file_loaded_with_prefix(self, tmp_path):
        profile = tmp_path / "doug.md"
        profile.write_text("Doug Robertson — Group Head of Responsible AI.\nReads for regulator-defensibility.")
        result = _resolve_persona(str(profile))
        assert result is not None
        assert result.startswith(PERSONA_PREFIX)
        assert "Doug Robertson" in result
        assert "regulator-defensibility" in result

    def test_missing_file_exits(self, tmp_path):
        nonexistent = tmp_path / "does-not-exist.md"
        with pytest.raises(SystemExit):
            _resolve_persona(str(nonexistent))


class TestMergePersonaContext:
    def test_both_none(self):
        assert _merge_persona_context(None, None) is None

    def test_persona_only(self):
        assert _merge_persona_context("PERSONA: foo", None) == "PERSONA: foo"

    def test_context_only(self):
        assert _merge_persona_context(None, "paper text") == "paper text"

    def test_persona_prepended(self):
        result = _merge_persona_context("PERSONA: foo", "paper text")
        assert result is not None
        assert result.startswith("PERSONA: foo")
        assert "paper text" in result
        # persona must come before paper
        assert result.index("PERSONA: foo") < result.index("paper text")
