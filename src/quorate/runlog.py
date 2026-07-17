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
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.5": (2.50, 10.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (15.00, 75.00),
    "grok-4.5": (2.00, 6.00),
    "grok-4.3": (3.00, 15.00),
    "grok-4.20-0309-reasoning": (3.00, 15.00),
    "kimi-k2.6": (0.66, 3.41),
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
    outcome: str | None = None
    outcome_note: str | None = None

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
            "outcome": self.outcome,
            "outcome_note": self.outcome_note,
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


def build_record(
    mode: str,
    results: Iterable[ModelCallResult],
    total_duration_s: float,
    judge_model: str | None = None,
    judge_result: ModelCallResult | None = None,
    extra_results: Iterable[ModelCallResult] | None = None,
    outcome: str | None = None,
    outcome_note: str | None = None,
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
        outcome=outcome,
        outcome_note=outcome_note,
    )


def _parse_outcome(line: str) -> tuple[str | None, str | None]:
    """Parse a one-line outcome reply into (tag, note).

    First token m/i selects matched/inverted; anything else (blank, s, ?) =>
    untagged. Text after the first token is kept as a free-text note.
    """
    line = (line or "").strip()
    if not line:
        return None, None
    head, _, rest = line.partition(" ")
    tag = {"m": "matched", "i": "inverted"}.get(head[:1].lower())
    return tag, (rest.strip() or None)


def prompt_outcome() -> tuple[str | None, str | None]:
    """Ask whether the verdict matched or inverted the going-in prior.

    TTY-guarded and exception-safe: returns (None, None) for non-interactive
    runs (dispatched/background/piped) so a council never blocks or crashes on
    stdin. The one bit this captures — matched vs inverted — is what makes the
    "is deliberation worth it?" question answerable from the run log.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None, None
    try:
        reply = input(
            "\nDid the verdict match or invert your going-in position? "
            "[m]atched / [i]nverted / [Enter] skip (+ optional note)\n> "
        )
    except (EOFError, KeyboardInterrupt):
        return None, None
    return _parse_outcome(reply)


def append(record: RunRecord, path: Path | None = None) -> Path:
    """Append a RunRecord as one JSON line. Returns the path written."""
    target = path or _default_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return target


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, ceil(0.95 * len(ordered)) - 1)]


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
    modes = Counter(str(record.get("mode", "unknown")) for record in records)
    for record in records:
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
    report = {
        "schema_version": 1,
        "generated_at": end.isoformat(),
        "window_days": days,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "runs": len(records),
        "modes": dict(sorted(modes.items())),
        "estimated_api_cost_usd": round(
            sum(float(record.get("est_cost_usd") or 0) for record in records), 4
        ),
        "models": model_rows,
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
