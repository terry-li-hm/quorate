"""Tests for quorate.runlog — JSONL persistence + cost estimation."""

from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path

import pytest

from quorate import runlog
from quorate.config import ModelCallResult
from quorate.divergence import format_divergence, parse_divergence


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

    def test_current_grok_and_glm_seats_have_explicit_prices(self):
        assert runlog.price_for("x-ai/grok-4.7") == (2.00, 6.00)
        assert runlog.price_for("z-ai/glm-5.3") == (0.0699, 1.40)

    def test_minimax_m3(self):
        assert runlog.price_for("minimax/minimax-m3") == (0.30, 1.20)

    def test_fireworks_kimi_k3_us_includes_us_priority_premium(self):
        assert runlog.price_for("kimi-k3-us") == (4.125, 20.625)

    def test_claude_5_family_prices_are_explicit(self):
        # Opus 5 has no prefix overlap with the 4.x rows, so an explicit row keeps
        # OpenRouter-fallback cost estimates off $0.
        assert runlog.price_for("anthropic/claude-opus-5") == (5.00, 25.00)
        # Opus 5.5 shares the claude-opus-5 prefix, so its own row must win.
        assert runlog.price_for("anthropic/claude-opus-5-5") == (4.00, 20.00)
        assert runlog.price_for("claude-fable-5-1") == (10.00, 50.00)
        assert runlog.price_for("claude-fable-5-2") == (8.00, 40.00)
        assert runlog.price_for("anthropic/claude-fable-5-2") == (8.00, 40.00)
        assert runlog.price_for("claude-sonnet-5") == (2.00, 10.00)


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
        [
            "codex-exec",
            "claude-print",
            "antigravity-cli",
            "gemini-cli",
            "kimi-code",
            "kimi-code-api",
        ],
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
        assert len(d["model_usage"]) == 2

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
            judge_model="anthropic/claude-fable-5",
            outcome="inverted",
            decision_value="improved",
            k3_effect="positive",
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
        assert report["model_usage"] == [
            {
                "name": "K3",
                "model_id": "k3",
                "calls": 1,
                "reachable": 1,
                "tokens_in": 100,
                "tokens_out": 200,
                "est_cost_usd": 0.0,
                "success_rate": 1.0,
                "mean_latency_s": 12.0,
                "p95_latency_s": 12.0,
                "providers": {"kimi-code": 1},
                "diagnostics": [],
            }
        ]
        assert report["natural_model_usage"] == report["model_usage"]
        assert report["evaluations"] == {
            "rated_runs": 1,
            "outcome": {"inverted": 1},
            "decision_value": {"improved": 1},
            "k3_effect": {"positive": 1},
            "decision_value_by_judge": {
                "anthropic/claude-fable-5": {"improved": 1},
            },
        }
        assert Path(report["snapshot_path"]).exists()

    def test_rejects_non_positive_window(self):
        with pytest.raises(ValueError, match="at least 1"):
            runlog.usage_report(0)

    def test_attributes_paid_extra_calls_to_model_and_route(self, tmp_path):
        log = tmp_path / "runs.jsonl"
        now = dt.datetime(2026, 7, 29, tzinfo=dt.timezone.utc)
        blind = _result(
            name="Kimi-K3-US",
            model_id="kimi-k3-us",
            provider="fireworks-us",
            tokens_in=1_000_000,
            tokens_out=1_000_000,
        )
        debate = _result(
            name="Kimi-K3-US",
            model_id="kimi-k3-us",
            provider="fireworks-us",
            tokens_in=2_000_000,
            tokens_out=2_000_000,
        )
        record = runlog.build_record(
            "council",
            [blind],
            total_duration_s=10,
            extra_results=[debate],
        ).to_dict()
        record["ts"] = now.isoformat()
        log.write_text(json.dumps(record) + "\n")

        report = runlog.usage_report(30, path=log, now=now)

        assert report["model_usage"] == [
            {
                "name": "Kimi-K3-US",
                "model_id": "kimi-k3-us",
                "calls": 2,
                "reachable": 2,
                "tokens_in": 3_000_000,
                "tokens_out": 3_000_000,
                "est_cost_usd": pytest.approx(74.25),
                "success_rate": 1.0,
                "mean_latency_s": 3.2,
                "p95_latency_s": 3.2,
                "providers": {"fireworks-us": 2},
                "diagnostics": [],
            }
        ]

    def test_extra_phase_failure_reduces_route_usage_success_rate(self):
        blind = _result(
            name="Kimi-K3-US",
            model_id="kimi-k3-us",
            provider="fireworks-us",
        )
        debate = _result(
            name="Kimi-K3-US",
            model_id="kimi-k3-us",
            provider="none",
            response="[Error: route unavailable]",
            tokens_in=None,
            tokens_out=None,
            diagnostics=("fireworks-us:http_503",),
        )

        record = runlog.build_record(
            "council",
            [blind],
            total_duration_s=10,
            extra_results=[debate],
        ).to_dict()

        assert record["model_usage"] == [
            {
                "name": "Kimi-K3-US",
                "model_id": "kimi-k3-us",
                "calls": 2,
                "reachable": 1,
                "tokens_in": 100,
                "tokens_out": 200,
                "est_cost_usd": pytest.approx(0.0045),
                "latencies_s": [3.2, 3.2],
                "providers": {"fireworks-us": 1, "none": 1},
                "diagnostics": ["fireworks-us:http_503"],
            }
        ]

    def test_natural_usage_excludes_synthetic_benchmark_calls(self, tmp_path):
        log = tmp_path / "runs.jsonl"
        now = dt.datetime(2026, 7, 29, tzinfo=dt.timezone.utc)
        benchmark_result = _result(
            name="Kimi-K3-US",
            model_id="kimi-k3-us",
            provider="fireworks-us",
        )
        natural_failure = _result(
            name="Kimi-K3-US",
            model_id="kimi-k3-us",
            provider="none",
            response="[Error: route unavailable]",
            tokens_in=None,
            tokens_out=None,
            diagnostics=("fireworks-us:http_503",),
        )
        benchmark_record = runlog.build_record(
            "benchmark",
            [benchmark_result],
            total_duration_s=1,
        ).to_dict()
        natural_record = runlog.build_record(
            "council",
            [natural_failure],
            total_duration_s=1,
        ).to_dict()
        benchmark_record["ts"] = now.isoformat()
        natural_record["ts"] = now.isoformat()
        log.write_text("\n".join((json.dumps(benchmark_record), json.dumps(natural_record))) + "\n")

        report = runlog.usage_report(30, path=log, now=now)

        assert report["model_usage"][0]["calls"] == 2
        assert report["model_usage"][0]["success_rate"] == 0.5
        assert report["natural_model_usage"][0]["calls"] == 1
        assert report["natural_model_usage"][0]["success_rate"] == 0.0
        assert report["natural_model_usage"][0]["diagnostics"] == ["fireworks-us:http_503"]


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
    def test_parse_choice_is_categorical_only(self):
        assert runlog._parse_choice("M", {"m": "matched"}) == "matched"
        assert runlog._parse_choice("m private rationale", {"m": "matched"}) == "matched"
        assert runlog._parse_choice("", {"m": "matched"}) is None
        assert runlog._parse_choice("s", {"m": "matched"}) is None

    def test_record_serialises_outcome(self):
        rec = runlog.build_record(
            "council",
            [_result()],
            total_duration_s=1.0,
            outcome="inverted",
            decision_value="improved",
            k3_effect="positive",
        )
        d = rec.to_dict()
        assert d["outcome"] == "inverted"
        assert d["decision_value"] == "improved"
        assert d["k3_effect"] == "positive"
        assert "outcome_note" not in d

    def test_default_outcome_is_null(self):
        rec = runlog.build_record("quick", [_result()], total_duration_s=1.0)
        d = rec.to_dict()
        assert d["outcome"] is None
        assert d["decision_value"] is None
        assert d["k3_effect"] is None

    def test_prompt_outcome_tty_branch(self, monkeypatch):
        # the capture path: interactive TTY + a real reply must parse through
        import builtins

        monkeypatch.setattr(runlog.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(runlog.sys.stdout, "isatty", lambda: True)
        replies = iter(("i", "b", "p"))
        monkeypatch.setattr(builtins, "input", lambda *a: next(replies))
        assert runlog.prompt_outcome(k3_present=True) == (
            "inverted",
            "improved",
            "positive",
        )

    def test_prompt_outcome_skips_k3_when_absent(self, monkeypatch):
        import builtins

        monkeypatch.setattr(runlog.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(runlog.sys.stdout, "isatty", lambda: True)
        replies = iter(("m", "s"))
        monkeypatch.setattr(builtins, "input", lambda *a: next(replies))
        assert runlog.prompt_outcome(k3_present=False) == (
            "matched",
            "unchanged",
            None,
        )

    def test_prompt_outcome_eof_mid_prompt_is_safe(self, monkeypatch):
        # Ctrl-D / EOF at the prompt must not crash a finished council
        import builtins

        monkeypatch.setattr(runlog.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(runlog.sys.stdout, "isatty", lambda: True)

        def _eof(*a):
            raise EOFError

        monkeypatch.setattr(builtins, "input", _eof)
        assert runlog.prompt_outcome() == (None, None, None)


def test_usage_report_rolls_up_divergence_by_decision_value(tmp_path):
    path = tmp_path / "runs.jsonl"
    now = dt.datetime(2026, 8, 8, tzinfo=dt.timezone.utc)
    rows = [
        {
            "ts": now.isoformat(),
            "mode": "council",
            "models": [],
            "decision_value": "improved",
            "divergence": {"index": 60.0, "distinct_positions": 3},
        },
        {
            "ts": now.isoformat(),
            "mode": "council",
            "models": [],
            "decision_value": "unchanged",
            "divergence": {"index": 10.0, "distinct_positions": 1},
        },
        {
            "ts": now.isoformat(),
            "mode": "council",
            "models": [],
            "decision_value": "unchanged",
            "divergence": {"index": 20.0, "distinct_positions": 1},
        },
        {"ts": now.isoformat(), "mode": "council", "models": [], "divergence": None},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = runlog.usage_report(days=7, path=path, now=now)
    summary = report["divergence"]

    assert summary["scored_runs"] == 3
    assert summary["mean_index"] == 30.0
    assert summary["min_index"] == 10.0 and summary["max_index"] == 60.0
    assert summary["distinct_positions"] == {"1": 2, "3": 1}
    assert summary["mean_index_by_decision_value"]["improved"] == {"runs": 1, "mean_index": 60.0}
    assert summary["mean_index_by_decision_value"]["unchanged"] == {"runs": 2, "mean_index": 15.0}


LABEL = "The fictitious cobalt goose prefers winter."
SENTINELS = [LABEL, "SECRET_PROMPT", "SECRET_NESTED", "SECRET_ANSWER"]


@pytest.fixture
def grouping():
    names = ["A", "B", "C", "D"]
    results = [_result(n, f"seat-{n}", response="SECRET_ANSWER") for n in names]
    groups = [{"label": LABEL, "members": names[:3]}, {"label": LABEL, "members": names[3:]}]
    live = parse_divergence(json.dumps({"positions": groups}), names)
    assert live is not None and LABEL in format_divergence(live)
    live["prompt"] = "SECRET_PROMPT"
    live["positions"][0]["extra"] = {"nested": ["SECRET_NESTED"]}
    return results, live


def test_q4_durable_projection_and_historical_rollup(tmp_path, grouping):
    results, live = grouping
    original = copy.deepcopy(live)
    path = tmp_path / "runs.jsonl"
    record = runlog.build_record("council", results, 1, divergence=live)
    durable = {
        "index": 25.0,
        "distinct_positions": 2,
        "largest_bloc": 3,
        "panel_size": 4,
        "positions": [{"members": ["seat-A", "seat-B", "seat-C"]}, {"members": ["seat-D"]}],
    }
    assert record.to_dict()["divergence"] == durable
    assert runlog.RunRecord(**record.to_dict()).to_dict()["divergence"] == durable
    runlog.append(record, path)
    raw = path.read_bytes()
    assert all(s.encode() not in raw for s in [*SENTINELS, '"label"'])
    assert json.loads(raw)["divergence"] == durable
    assert live == original

    record.divergence = None
    runlog.append(record, path)
    historical = record.to_dict()
    historical["divergence"] = parse_divergence(
        json.dumps(
            {
                "positions": [
                    {"members": ["A", "B"], "label": LABEL},
                    {"members": ["C", "D"], "label": LABEL},
                ]
            }
        ),
        ["A", "B", "C", "D"],
    )
    with path.open("a") as f:
        f.write(json.dumps(historical) + "\n")
    report = runlog.usage_report(path=path)
    assert LABEL not in json.dumps(report)
    assert report["runs"] == 3
    assert report["divergence"] == {
        "scored_runs": 2,
        "mean_index": 37.5,
        "min_index": 25.0,
        "max_index": 50.0,
        "distinct_positions": {"2": 2},
        "mean_index_by_decision_value": {},
    }


@pytest.mark.parametrize(
    "change",
    [
        None,
        {"index": "25"},
        {"index": float("nan")},
        {"index": float("inf")},
        {"index": 10**400},
        {"index": 0.0},
        {"panel_size": True},
        {"largest_bloc": 3.0},
        {"distinct_positions": 1},
        {"positions": {}},
        {"positions": [None]},
        {"positions": [{"members": [[]]}]},
        {"positions": [{"members": [LABEL]}]},
        {"positions": [{"members": ["A", "B", "C"]}, {"members": ["A"]}]},
        {"positions": [{"members": ["A", "B", "C"]}]},
    ],
)
def test_q4_durable_invalid_assignment(tmp_path, grouping, change):
    results, live = grouping
    record = runlog.build_record("council", results, 1)
    record.divergence = None if change is None else live | change
    raw = runlog.append(record, tmp_path / "runs").read_bytes()
    assert json.loads(raw)["divergence"] is None
    assert all(s.encode() not in raw for s in SENTINELS)


def test_q4_durable_requires_successful_seats(tmp_path, grouping):
    results, live = grouping
    record = runlog.RunRecord(**runlog.build_record("council", results, 1).to_dict())
    record.divergence = live
    path = tmp_path / "runs"
    runlog.append(record, path)
    assert json.loads(path.read_bytes())["divergence"]["index"] == 25.0
    record.models[-1]["ok"] = False
    runlog.append(record, path)
    raw = path.read_bytes()
    assert json.loads(raw.splitlines()[1])["divergence"] is None
    assert all(s.encode() not in raw for s in SENTINELS)


@pytest.mark.parametrize("include_valid_seats", [False, True])
def test_q4_durable_malformed_roster_row(tmp_path, grouping, include_valid_seats):
    results, live = grouping
    fields = runlog.build_record("council", results, 1).to_dict()
    fields["models"] = [*fields["models"], None] if include_valid_seats else [None]
    fields["divergence"] = live
    record = runlog.RunRecord(**fields)
    assert record.to_dict()["divergence"] is None
    raw = runlog.append(record, tmp_path / "runs").read_bytes()
    assert json.loads(raw)["divergence"] is None
    assert all(s.encode() not in raw for s in SENTINELS)
