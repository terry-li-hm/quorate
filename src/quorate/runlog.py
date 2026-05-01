"""Run log — JSONL persistence and cost estimation for quorate runs."""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from quorate.config import ModelCallResult


def _default_log_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "quorate" / "runs.jsonl"


PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.1-pro-preview": (1.25, 10.00),
    "gpt-5.5":                (2.50, 10.00),
    "claude-opus-4-7":        (15.00, 75.00),
    "grok-4.3":               (3.00, 15.00),
    "grok-4.20-0309-reasoning": (3.00, 15.00),
    "kimi-k2.6":              (0.60, 2.50),
    "glm-5.1":                (0.0, 0.0),
    "mimo-v2.5-pro":          (0.0, 0.0),
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
        }


def _model_row(result: ModelCallResult) -> dict:
    return {
        "name": result.name,
        "model_id": result.model_id,
        "duration_s": round(result.latency_s, 2),
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "ok": not result.is_error,
    }


def build_record(
    mode: str,
    results: Iterable[ModelCallResult],
    total_duration_s: float,
    judge_model: str | None = None,
    judge_result: ModelCallResult | None = None,
    extra_results: Iterable[ModelCallResult] | None = None,
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
    )


def append(record: RunRecord, path: Path | None = None) -> Path:
    """Append a RunRecord as one JSON line. Returns the path written."""
    target = path or _default_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return target


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
    summary = f"({total_duration_s:.1f}s) — ${total_cost:.4f} (in: ${in_cost:.4f}, out: ${out_cost:.4f})"
    return lines, summary
