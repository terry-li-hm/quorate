"""Tests for quorate.runlog — JSONL persistence + cost estimation."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from quorate import runlog
from quorate.config import ModelCallResult


def _result(
    name: str = "Grok-4.3",
    model_id: str = "grok-4.3",
    tokens_in: int | None = 100,
    tokens_out: int | None = 200,
    latency: float = 3.2,
    response: str = "ok",
    provider: str = "xai-native",
    diagnostics: tuple[str, ...] = (),
) -> ModelCallResult:
    return ModelCallResult(
        name=name,
        model_id=model_id,
        response=response,
        provider=provider,
        latency_s=latency,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        diagnostics=diagnostics,
    )


class TestPriceLookup:
    def test_known_grok43(self):
        in_p, out_p = runlog.price_for("grok-4.3")
        assert in_p > 0 and out_p > 0

    def test_x_ai_prefix(self):
        in_p, out_p = runlog.price_for("x-ai/grok-4.3")
        assert (in_p, out_p) == runlog.price_for("grok-4.3")

    def test_unknown_returns_zero(self):
        assert runlog.price_for("totally-fake-model-9000") == (0.0, 0.0)

    def test_zhipu_glm51_free(self):
        assert runlog.price_for("glm-5.1") == (0.0, 0.0)

    def test_minimax_m3(self):
        assert runlog.price_for("minimax/minimax-m3") == (0.30, 1.20)


class TestEstimateCost:
    def test_grok43_basic(self):
        r = _result(model_id="grok-4.3", tokens_in=1_000_000, tokens_out=1_000_000)
        in_p, out_p = runlog.price_for("grok-4.3")
        assert runlog.estimate_cost(r) == pytest.approx(in_p + out_p)

    def test_zero_tokens(self):
        r = _result(tokens_in=0, tokens_out=0)
        assert runlog.estimate_cost(r) == 0.0

    def test_none_tokens(self):
        r = _result(tokens_in=None, tokens_out=None)
        assert runlog.estimate_cost(r) == 0.0

    def test_unknown_model_zero(self):
        r = _result(model_id="fake-model", tokens_in=1000, tokens_out=1000)
        assert runlog.estimate_cost(r) == 0.0

    @pytest.mark.parametrize(
        "provider",
        ["codex-exec", "claude-print", "antigravity-cli", "gemini-cli", "kimi-code"],
    )
    def test_subscription_routes_have_zero_marginal_cost(self, provider):
        r = _result(
            model_id="claude-fable-5",
            tokens_in=1_000_000,
            tokens_out=1_000_000,
            provider=provider,
        )
        assert runlog.estimate_cost(r) == 0.0


class TestBuildRecord:
    def test_basic(self):
        results = [
            _result(),
            _result(name="GPT-5.5", model_id="gpt-5.5", tokens_in=50, tokens_out=80),
        ]
        record = runlog.build_record("quick", results, total_duration_s=12.5)
        d = record.to_dict()
        assert d["mode"] == "quick"
        assert len(d["models"]) == 2
        assert d["total_tokens_in"] == 150
        assert d["total_tokens_out"] == 280
        assert d["total_duration_s"] == 12.5
        assert d["est_cost_usd"] > 0

    def test_failed_model_marked(self):
        bad = _result(
            response="[Error: All providers failed]",
            provider="none",
            diagnostics=("xai-native:http_404", "openrouter:http_403"),
        )
        record = runlog.build_record("quick", [bad], total_duration_s=1.0)
        assert record.models[0]["ok"] is False
        assert record.models[0]["provider"] == "none"
        assert record.models[0]["diagnostics"] == [
            "xai-native:http_404",
            "openrouter:http_403",
        ]


class TestAppend:
    def test_append_creates_dir_and_file(self, tmp_path):
        log = tmp_path / "subdir" / "runs.jsonl"
        record = runlog.build_record("quick", [_result()], total_duration_s=3.2)
        written = runlog.append(record, path=log)
        assert written == log
        assert log.exists()
        line = log.read_text().strip()
        parsed = json.loads(line)
        assert parsed["mode"] == "quick"

    def test_append_is_additive(self, tmp_path):
        log = tmp_path / "runs.jsonl"
        record1 = runlog.build_record("quick", [_result()], total_duration_s=1.0)
        record2 = runlog.build_record("council", [_result()], total_duration_s=2.0)
        runlog.append(record1, path=log)
        runlog.append(record2, path=log)
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["mode"] == "quick"
        assert json.loads(lines[1])["mode"] == "council"


class TestUsageReport:
    def test_aggregates_window_and_saves_snapshot(self, tmp_path):
        log = tmp_path / "runs.jsonl"
        now = dt.datetime(2026, 7, 17, tzinfo=dt.timezone.utc)
        recent = runlog.build_record(
            "council",
            [
                _result(
                    name="K3",
                    model_id="k3",
                    provider="kimi-code",
                    latency=12.0,
                )
            ],
            total_duration_s=20,
        ).to_dict()
        recent["ts"] = (now - dt.timedelta(days=2)).isoformat()
        old = dict(recent)
        old["ts"] = (now - dt.timedelta(days=40)).isoformat()
        log.write_text("\n".join((json.dumps(recent), "not-json", json.dumps(old))) + "\n")

        report = runlog.usage_report(
            30,
            path=log,
            now=now,
            save=True,
            snapshot_dir=tmp_path / "usage",
        )

        assert report["runs"] == 1
        assert report["models"][0]["model_id"] == "k3"
        assert report["models"][0]["success_rate"] == 1.0
        assert report["models"][0]["providers"] == {"kimi-code": 1}
        assert Path(report["snapshot_path"]).exists()

    def test_rejects_non_positive_window(self):
        with pytest.raises(ValueError, match="at least 1"):
            runlog.usage_report(0)


class TestFormatFooter:
    def test_lines_and_summary(self):
        results = [_result(), _result(name="GPT-5.5", model_id="gpt-5.5", latency=5.0)]
        lines, summary = runlog.format_footer(results, total_duration_s=8.5)
        assert len(lines) == 2
        assert "Grok-4.3" in lines[0]
        assert "GPT-5.5" in lines[1]
        assert "8.5s" in summary
        assert "$" in summary

    def test_failed_model_shown_fail(self):
        bad = _result(response="[Error: timeout]")
        lines, _ = runlog.format_footer([bad], total_duration_s=1.0)
        assert "FAIL" in lines[0]


class TestOutcome:
    def test_parse_matched_inverted_skip(self):
        assert runlog._parse_outcome("m") == ("matched", None)
        assert runlog._parse_outcome("i") == ("inverted", None)
        assert runlog._parse_outcome("") == (None, None)
        assert runlog._parse_outcome("s") == (None, None)

    def test_parse_keeps_note(self):
        assert runlog._parse_outcome("i flipped my call on precedent") == (
            "inverted",
            "flipped my call on precedent",
        )
        # case-insensitive head, note preserved
        assert runlog._parse_outcome("M confirmed prior") == ("matched", "confirmed prior")

    def test_record_serialises_outcome(self):
        rec = runlog.build_record(
            "council",
            [_result()],
            total_duration_s=1.0,
            outcome="inverted",
            outcome_note="changed the decision",
        )
        d = rec.to_dict()
        assert d["outcome"] == "inverted"
        assert d["outcome_note"] == "changed the decision"

    def test_default_outcome_is_null(self):
        rec = runlog.build_record("quick", [_result()], total_duration_s=1.0)
        d = rec.to_dict()
        assert d["outcome"] is None
        assert d["outcome_note"] is None

    def test_prompt_outcome_tty_branch(self, monkeypatch):
        # the capture path: interactive TTY + a real reply must parse through
        import builtins

        monkeypatch.setattr(runlog.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(runlog.sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(builtins, "input", lambda *a: "i changed my call")
        assert runlog.prompt_outcome() == ("inverted", "changed my call")

    def test_prompt_outcome_eof_mid_prompt_is_safe(self, monkeypatch):
        # Ctrl-D / EOF at the prompt must not crash a finished council
        import builtins

        monkeypatch.setattr(runlog.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(runlog.sys.stdout, "isatty", lambda: True)

        def _eof(*a):
            raise EOFError

        monkeypatch.setattr(builtins, "input", _eof)
        assert runlog.prompt_outcome() == (None, None)
