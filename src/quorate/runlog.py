"""Run log — JSONL persistence and cost estimation for quorate runs."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from statistics import mean
from typing import Iterable

from quorate.config import ModelCallResult


def _default_log_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "quorate" / "runs.jsonl"


def _default_usage_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "quorate" / "usage"


PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.5-flash": (1.50, 9.00),
    # List price; Google's introductory 2.00/10.00 ends on 30 November 2026.
    "gemini-4-argon": (4.00, 20.00),
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.5": (2.50, 10.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (15.00, 75.00),
    "grok-4.5": (2.00, 6.00),
    "grok-4.3": (3.00, 15.00),
    "grok-4.20-0309-reasoning": (3.00, 15.00),
    "kimi-k2.6": (0.66, 3.41),
    # Fireworks K3 US Priority: input/output USD per 1M tokens. Cached-input
    # pricing is not retained in ModelCallResult, so input remains conservative.
    "kimi-k3-us": (4.125, 20.625),
    "glm-5.2": (0.9576, 3.0096),
    "glm-5.1": (0.0, 0.0),
    "mimo-v2.5-pro": (0.435, 0.87),
    "deepseek-v4-pro": (0.435, 0.87),
    "minimax-m3": (0.30, 1.20),
}

ZERO_MARGINAL_COST_PROVIDERS = {
    "antigravity-cli",
    "claude-print",
    "codex-exec",
    "kimi-code",
    "kimi-code-api",
    "gemini-cli",  # Historical run records.
    "zhipu-native",
}


def price_for(model_id: str) -> tuple[float, float]:
    """Look up (input_per_1M, output_per_1M) for a model_id. Suffix match."""
    key = model_id.rsplit("/", 1)[-1].lower()
    if key in PRICES:
        return PRICES[key]
    for known, prices in PRICES.items():
        if key.startswith(known) or known.startswith(key):
            return prices
    return (0.0, 0.0)


def estimate_cost(result: ModelCallResult) -> float:
    """USD estimate for one ModelCallResult. Returns 0.0 if priced at 0 or no tokens."""
    if result.provider in ZERO_MARGINAL_COST_PROVIDERS:
        return 0.0
    if not result.tokens_in and not result.tokens_out:
        return 0.0
    in_price, out_price = price_for(result.model_id)
    cost = 0.0
    if result.tokens_in:
        cost += (result.tokens_in / 1_000_000) * in_price
    if result.tokens_out:
        cost += (result.tokens_out / 1_000_000) * out_price
    return cost


def split_cost(result: ModelCallResult) -> tuple[float, float]:
    """Returns (input_cost_usd, output_cost_usd) for a single result."""
    if result.provider in ZERO_MARGINAL_COST_PROVIDERS:
        return (0.0, 0.0)
    in_price, out_price = price_for(result.model_id)
    in_cost = ((result.tokens_in or 0) / 1_000_000) * in_price
    out_cost = ((result.tokens_out or 0) / 1_000_000) * out_price
    return in_cost, out_cost


@dataclass
class RunRecord:
    """One quorate run, serialised to JSONL."""

    ts: str
    mode: str
    models: list[dict]
    judge_model: str | None
    total_duration_s: float
    total_tokens_in: int
    total_tokens_out: int
    est_cost_usd: float
    model_usage: list[dict]
    outcome: str | None = None
    decision_value: str | None = None
    k3_effect: str | None = None
    divergence: dict | None = None

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "mode": self.mode,
            "models": self.models,
            "judge_model": self.judge_model,
            "total_duration_s": round(self.total_duration_s, 2),
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "est_cost_usd": round(self.est_cost_usd, 4),
            "model_usage": self.model_usage,
            "outcome": self.outcome,
            "decision_value": self.decision_value,
            "k3_effect": self.k3_effect,
            "divergence": self.divergence,
        }


def _model_row(result: ModelCallResult) -> dict:
    row = {
        "name": result.name,
        "model_id": result.model_id,
        "provider": result.provider,
        "duration_s": round(result.latency_s, 2),
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "ok": not result.is_error,
    }
    if result.diagnostics:
        row["diagnostics"] = list(result.diagnostics)
    return row


def _aggregate_model_usage(results: Iterable[ModelCallResult]) -> list[dict]:
    """Aggregate response-free token and cost telemetry by model and route."""
    aggregates: dict[str, dict] = {}
    for result in results:
        key = result.model_id
        row = aggregates.setdefault(
            key,
            {
                "name": result.name,
                "model_id": result.model_id,
                "calls": 0,
                "reachable": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "est_cost_usd": 0.0,
                "latencies_s": [],
                "providers": Counter(),
                "diagnostics": set(),
            },
        )
        row["calls"] += 1
        row["reachable"] += not result.is_error
        row["tokens_in"] += result.tokens_in or 0
        row["tokens_out"] += result.tokens_out or 0
        row["est_cost_usd"] += estimate_cost(result)
        row["latencies_s"].append(round(result.latency_s, 2))
        row["providers"][result.provider] += 1
        row["diagnostics"].update(result.diagnostics)
    rows = []
    for row in aggregates.values():
        rows.append(
            {
                **row,
                "est_cost_usd": round(row["est_cost_usd"], 4),
                "providers": dict(sorted(row["providers"].items())),
                "diagnostics": sorted(row["diagnostics"]),
            }
        )
    return sorted(rows, key=lambda row: str(row["model_id"]))


def build_record(
    mode: str,
    results: Iterable[ModelCallResult],
    total_duration_s: float,
    judge_model: str | None = None,
    judge_result: ModelCallResult | None = None,
    extra_results: Iterable[ModelCallResult] | None = None,
    outcome: str | None = None,
    decision_value: str | None = None,
    k3_effect: str | None = None,
    divergence: dict | None = None,
) -> RunRecord:
    """Assemble a RunRecord. extra_results carries judge/critique cost into totals."""
    results_list = list(results)
    rows = [_model_row(r) for r in results_list]
    all_results = list(results_list)
    if judge_result is not None:
        all_results.append(judge_result)
    if extra_results:
        all_results.extend(extra_results)
    total_in = sum((r.tokens_in or 0) for r in all_results)
    total_out = sum((r.tokens_out or 0) for r in all_results)
    cost = sum(estimate_cost(r) for r in all_results)
    return RunRecord(
        ts=dt.datetime.now(dt.timezone.utc).isoformat(),
        mode=mode,
        models=rows,
        judge_model=judge_model,
        total_duration_s=total_duration_s,
        total_tokens_in=total_in,
        total_tokens_out=total_out,
        est_cost_usd=cost,
        model_usage=_aggregate_model_usage(all_results),
        outcome=outcome,
        decision_value=decision_value,
        k3_effect=k3_effect,
        divergence=divergence,
    )


def _parse_choice(line: str, choices: dict[str, str]) -> str | None:
    """Parse one categorical key without retaining free text."""
    value = (line or "").strip().lower()
    return choices.get(value[:1]) if value else None


def prompt_outcome(*, k3_present: bool = False) -> tuple[str | None, str | None, str | None]:
    """Collect categorical alignment, value, and K3-effect signals.

    TTY-guarded and exception-safe: returns null signals for non-interactive
    runs (dispatched/background/piped) so a council never blocks or crashes on
    stdin. No prompt, response, rationale, or free-text note is retained.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None, None, None
    try:
        outcome = _parse_choice(
            input(
                "\nDid the verdict match or invert your going-in position? "
                "[m]atched / [i]nverted / [Enter] skip\n> "
            ),
            {"m": "matched", "i": "inverted"},
        )
        decision_value = _parse_choice(
            input(
                "Did the council improve the final decision? "
                "[b]etter / [s]ame / [w]orse / [p]ending / [Enter] skip\n> "
            ),
            {"b": "improved", "s": "unchanged", "w": "worsened", "p": "pending"},
        )
        k3_effect = None
        if k3_present:
            k3_effect = _parse_choice(
                input(
                    "Did K3 add distinct value? "
                    "[p]ositive / [n]eutral / [h]armful / [u]nclear / [Enter] skip\n> "
                ),
                {"p": "positive", "n": "neutral", "h": "negative", "u": "unclear"},
            )
    except (EOFError, KeyboardInterrupt):
        return None, None, None
    return outcome, decision_value, k3_effect


def append(record: RunRecord, path: Path | None = None) -> Path:
    """Append a RunRecord as one JSON line. Returns the path written."""
    target = path or _default_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return target


def _divergence_summary(records: Iterable[dict]) -> dict:
    """Roll blind-divergence scores up, split by how the run was later rated.

    The split is the point. A high mean tells you the panel started apart; the
    breakdown against decision_value tells you whether starting apart is what
    made the panel worth convening, which is the only way to learn from the
    log which question shapes deserve a council and which merely cost one.
    """
    indices: list[float] = []
    by_value: dict[str, list[float]] = {}
    positions: Counter = Counter()
    for record in records:
        divergence = record.get("divergence")
        if not isinstance(divergence, dict):
            continue
        index = divergence.get("index")
        if not isinstance(index, int | float) or isinstance(index, bool):
            continue
        indices.append(float(index))
        distinct = divergence.get("distinct_positions")
        if isinstance(distinct, int) and not isinstance(distinct, bool):
            positions[str(distinct)] += 1
        if record.get("decision_value"):
            by_value.setdefault(str(record["decision_value"]), []).append(float(index))
    return {
        "scored_runs": len(indices),
        "mean_index": round(mean(indices), 1) if indices else None,
        "min_index": round(min(indices), 1) if indices else None,
        "max_index": round(max(indices), 1) if indices else None,
        "distinct_positions": dict(sorted(positions.items())),
        "mean_index_by_decision_value": {
            value: {"runs": len(scores), "mean_index": round(mean(scores), 1)}
            for value, scores in sorted(by_value.items())
        },
    }


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, ceil(0.95 * len(ordered)) - 1)]


def _accumulate_usage_rows(aggregates: dict[str, dict], usage_rows: Iterable[dict]) -> None:
    """Merge response-free per-run model usage into rolling aggregates."""
    for usage in usage_rows:
        model_id = str(usage.get("model_id") or usage.get("name") or "unknown")
        aggregate = aggregates.setdefault(
            model_id,
            {
                "name": str(usage.get("name") or model_id),
                "model_id": model_id,
                "calls": 0,
                "reachable": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "est_cost_usd": 0.0,
                "latencies_s": [],
                "providers": Counter(),
                "diagnostics": set(),
            },
        )
        aggregate["calls"] += int(usage.get("calls") or 0)
        aggregate["reachable"] += int(usage.get("reachable") or 0)
        aggregate["tokens_in"] += int(usage.get("tokens_in") or 0)
        aggregate["tokens_out"] += int(usage.get("tokens_out") or 0)
        aggregate["est_cost_usd"] += float(usage.get("est_cost_usd") or 0)
        aggregate["latencies_s"].extend(float(value) for value in usage.get("latencies_s", []))
        aggregate["providers"].update(usage.get("providers") or {})
        aggregate["diagnostics"].update(usage.get("diagnostics") or ())


def _render_usage_rows(aggregates: dict[str, dict]) -> list[dict]:
    """Render rolling usage aggregates without exposing response content."""
    rows = []
    for aggregate in aggregates.values():
        latencies = list(aggregate["latencies_s"])
        calls = int(aggregate["calls"])
        reachable = int(aggregate["reachable"])
        rows.append(
            {
                "name": aggregate["name"],
                "model_id": aggregate["model_id"],
                "calls": calls,
                "reachable": reachable,
                "tokens_in": int(aggregate["tokens_in"]),
                "tokens_out": int(aggregate["tokens_out"]),
                "est_cost_usd": round(float(aggregate["est_cost_usd"]), 4),
                "success_rate": round(reachable / calls, 3) if calls else 0.0,
                "mean_latency_s": round(mean(latencies), 2) if latencies else None,
                "p95_latency_s": (
                    round(value, 2) if (value := _percentile_95(latencies)) is not None else None
                ),
                "providers": dict(sorted(aggregate["providers"].items())),
                "diagnostics": sorted(aggregate["diagnostics"]),
            }
        )
    return sorted(rows, key=lambda row: str(row["model_id"]))


def usage_report(
    days: int = 30,
    *,
    path: Path | None = None,
    now: dt.datetime | None = None,
    save: bool = False,
    snapshot_dir: Path | None = None,
) -> dict:
    """Aggregate non-sensitive route telemetry over a rolling window."""
    if days < 1:
        raise ValueError("days must be at least 1")
    end = now or dt.datetime.now(dt.timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)
    end = end.astimezone(dt.timezone.utc)
    start = end - dt.timedelta(days=days)
    target = path or _default_log_path()
    records: list[dict] = []
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                timestamp = dt.datetime.fromisoformat(str(record["ts"]).replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
                timestamp = timestamp.astimezone(dt.timezone.utc)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if start <= timestamp <= end:
                records.append(record)

    aggregates: dict[str, dict] = {}
    usage_aggregates: dict[str, dict] = {}
    natural_usage_aggregates: dict[str, dict] = {}
    modes = Counter(str(record.get("mode", "unknown")) for record in records)
    outcomes = Counter(str(record["outcome"]) for record in records if record.get("outcome"))
    decision_values = Counter(
        str(record["decision_value"]) for record in records if record.get("decision_value")
    )
    k3_effects = Counter(str(record["k3_effect"]) for record in records if record.get("k3_effect"))
    judge_values: dict[str, Counter] = {}
    for record in records:
        if record.get("judge_model") and record.get("decision_value"):
            judge = str(record["judge_model"])
            judge_values.setdefault(judge, Counter())[str(record["decision_value"])] += 1
        for model in record.get("models", []):
            model_id = str(model.get("model_id") or model.get("name") or "unknown")
            aggregate = aggregates.setdefault(
                model_id,
                {
                    "name": str(model.get("name") or model_id),
                    "model_id": model_id,
                    "appearances": 0,
                    "reachable": 0,
                    "durations": [],
                    "providers": Counter(),
                },
            )
            aggregate["appearances"] += 1
            aggregate["reachable"] += bool(model.get("ok"))
            duration = model.get("duration_s")
            if isinstance(duration, int | float):
                aggregate["durations"].append(float(duration))
            aggregate["providers"][str(model.get("provider") or "unknown")] += 1
        usage_rows = record.get("model_usage")
        if not isinstance(usage_rows, list):
            usage_rows = []
            for model in record.get("models", []):
                model_id = str(model.get("model_id") or model.get("name") or "unknown")
                provider = str(model.get("provider") or "unknown")
                result = ModelCallResult(
                    name=str(model.get("name") or model_id),
                    model_id=model_id,
                    response="ok" if model.get("ok") else "[Error: unavailable]",
                    provider=provider,
                    latency_s=float(model.get("duration_s") or 0),
                    tokens_in=model.get("tokens_in"),
                    tokens_out=model.get("tokens_out"),
                    diagnostics=tuple(model.get("diagnostics") or ()),
                )
                usage_rows.extend(_aggregate_model_usage([result]))
        _accumulate_usage_rows(usage_aggregates, usage_rows)
        if str(record.get("mode") or "unknown") != "benchmark":
            _accumulate_usage_rows(natural_usage_aggregates, usage_rows)

    model_rows = []
    for aggregate in aggregates.values():
        durations = aggregate.pop("durations")
        appearances = int(aggregate["appearances"])
        reachable = int(aggregate["reachable"])
        model_rows.append(
            {
                **aggregate,
                "success_rate": round(reachable / appearances, 3) if appearances else 0.0,
                "mean_latency_s": round(mean(durations), 2) if durations else None,
                "p95_latency_s": (
                    round(value, 2) if (value := _percentile_95(durations)) is not None else None
                ),
                "providers": dict(sorted(aggregate["providers"].items())),
            }
        )
    model_rows.sort(key=lambda row: (-int(row["appearances"]), str(row["name"])))
    model_usage = _render_usage_rows(usage_aggregates)
    natural_model_usage = _render_usage_rows(natural_usage_aggregates)
    report = {
        "schema_version": 5,
        "generated_at": end.isoformat(),
        "window_days": days,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "runs": len(records),
        "modes": dict(sorted(modes.items())),
        "evaluations": {
            "rated_runs": sum(decision_values.values()),
            "outcome": dict(sorted(outcomes.items())),
            "decision_value": dict(sorted(decision_values.items())),
            "k3_effect": dict(sorted(k3_effects.items())),
            "decision_value_by_judge": {
                judge: dict(sorted(values.items()))
                for judge, values in sorted(judge_values.items())
            },
        },
        "divergence": _divergence_summary(records),
        "estimated_api_cost_usd": round(
            sum(float(record.get("est_cost_usd") or 0) for record in records), 4
        ),
        "models": model_rows,
        "model_usage": model_usage,
        "natural_model_usage": natural_model_usage,
        "source": str(target),
    }
    if save:
        directory = snapshot_dir or _default_usage_dir()
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = directory / f"{end.date().isoformat()}.json"
        snapshot.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report["snapshot_path"] = str(snapshot)
    return report


def format_footer(
    results: list[ModelCallResult],
    total_duration_s: float,
    extra_results: list[ModelCallResult] | None = None,
) -> tuple[list[str], str]:
    """Return (per_model_lines, summary_line). Caller wraps in [dim]...[/dim]."""
    lines: list[str] = []
    name_w = max((len(r.name) for r in results), default=10)
    for r in results:
        status = "ok" if not r.is_error else "FAIL"
        tok_out = r.tokens_out if r.tokens_out is not None else "-"
        lines.append(f"  {r.name:<{name_w}}  {r.latency_s:>6.2f}s  out={tok_out!s:<6}  {status}")
    all_results = list(results) + list(extra_results or [])
    in_cost = sum(split_cost(r)[0] for r in all_results)
    out_cost = sum(split_cost(r)[1] for r in all_results)
    total_cost = in_cost + out_cost
    summary = (
        f"({total_duration_s:.1f}s) — ${total_cost:.4f} (in: ${in_cost:.4f}, out: ${out_cost:.4f})"
    )
    return lines, summary
