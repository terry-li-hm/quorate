"""Synthetic roster canaries and monthly benchmark snapshots."""

from __future__ import annotations

import datetime as dt
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

from quorate.api import run_parallel
from quorate.config import Message, ModelCallResult, ModelEntry, ReasoningEffort, benchmark_models

SUITE_VERSION = "2026-07-16"


@dataclass(frozen=True)
class Canary:
    """A fixed synthetic prompt with a deterministic response validator."""

    id: str
    prompt: str
    validator: Callable[[str], bool]


def _exact(expected: str) -> Callable[[str], bool]:
    return lambda response: response.strip() == expected


def _valid_json_contract(response: str) -> bool:
    try:
        parsed = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError):
        return False
    return parsed == {"status": "ok", "items": [1, 2, 3]}


CANARIES: tuple[Canary, ...] = (
    Canary(
        id="exact-token",
        prompt=("Synthetic Quorate route canary. Reply with exactly QUORATE_OK and nothing else."),
        validator=_exact("QUORATE_OK"),
    ),
    Canary(
        id="json-contract",
        prompt=(
            "Synthetic Quorate structure canary. Return exactly this JSON object with no markdown: "
            '{"status":"ok","items":[1,2,3]}'
        ),
        validator=_valid_json_contract,
    ),
    Canary(
        id="simple-reasoning",
        prompt=(
            "Synthetic Quorate reasoning canary. Compute nine multiplied by five, then subtract "
            "three. Reply with only the integer result."
        ),
        validator=_exact("42"),
    ),
)


def default_snapshot_dir() -> Path:
    """Return the local state directory for dated benchmark snapshots."""
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "quorate" / "benchmarks"


def _attempt_row(canary: Canary, result: ModelCallResult) -> dict[str, Any]:
    reachable = not result.is_error
    passed = reachable and canary.validator(result.response)
    row: dict[str, Any] = {
        "canary": canary.id,
        "name": result.name,
        "model_id": result.model_id,
        "provider": result.provider,
        "reachable": reachable,
        "passed": passed,
        "latency_s": round(result.latency_s, 2),
    }
    if result.diagnostics:
        row["diagnostics"] = list(result.diagnostics)
    return row


def _summarize_models(
    models: Sequence[ModelEntry],
    attempts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for entry in models:
        rows = [row for row in attempts if row["name"] == entry.name]
        providers = Counter(str(row["provider"]) for row in rows)
        diagnostics = sorted(
            {diagnostic for row in rows for diagnostic in row.get("diagnostics", [])}
        )
        passed = sum(bool(row["passed"]) for row in rows)
        reachable = sum(bool(row["reachable"]) for row in rows)
        summary: dict[str, Any] = {
            "name": entry.name,
            "model_id": entry.model,
            "canaries": len(rows),
            "reachable": reachable,
            "passed": passed,
            "pass_rate": round(passed / len(rows), 3) if rows else 0.0,
            "mean_latency_s": round(mean(float(row["latency_s"]) for row in rows), 2)
            if rows
            else 0.0,
            "providers": dict(sorted(providers.items())),
        }
        if diagnostics:
            summary["diagnostics"] = diagnostics
        summaries.append(summary)
    return summaries


def save_report(report: dict[str, Any], directory: Path | None = None) -> Path:
    """Write one idempotent snapshot per UTC date."""
    target_dir = directory or default_snapshot_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    date = str(report["ts"])[:10]
    target = target_dir / f"{date}.json"
    report["snapshot_path"] = str(target)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


async def run_benchmark(
    *,
    models: list[ModelEntry] | None = None,
    canaries: Sequence[Canary] = CANARIES,
    timeout: float = 90,
    save: bool = True,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Run fixed synthetic canaries and return a roster health report.

    The report deliberately excludes model response text. It records only route,
    latency, deterministic pass state, and safe diagnostics.
    """
    resolved_models = models or benchmark_models()
    attempts: list[dict[str, Any]] = []
    canary_summaries: list[dict[str, Any]] = []
    quorum_target = max(2, len(resolved_models) // 2 + 1)

    for canary in canaries:
        results = await run_parallel(
            resolved_models,
            [Message.user(canary.prompt)],
            max_tokens=256,
            timeout=max(timeout, 180),
            effort=ReasoningEffort.MEDIUM,
        )
        rows = [_attempt_row(canary, result) for result in results]
        attempts.extend(rows)
        reachable_count = sum(bool(row["reachable"]) for row in rows)
        pass_count = sum(bool(row["passed"]) for row in rows)
        canary_summaries.append(
            {
                "id": canary.id,
                "reachable_count": reachable_count,
                "pass_count": pass_count,
                "quorum_target": quorum_target,
                "quorum_achieved": pass_count >= quorum_target,
            }
        )

    model_summaries = _summarize_models(resolved_models, attempts)
    quorum_achieved = all(row["quorum_achieved"] for row in canary_summaries)
    weak_seats = [row["name"] for row in model_summaries if row["pass_rate"] < 2 / 3]
    if not quorum_achieved:
        status = "unhealthy"
        action = "Inspect failed routes before relying on Quorate; do not change the roster yet."
    elif len(weak_seats) > 1:
        status = "degraded"
        action = (
            "Review the weak seats against external benchmarks; "
            "roster changes require both signals."
        )
    else:
        status = "healthy"
        action = "Keep the roster. Reconsider only when external and local evidence agree."

    report: dict[str, Any] = {
        "schema_version": 1,
        "suite_version": SUITE_VERSION,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "quorum_achieved": quorum_achieved,
        "models_total": len(resolved_models),
        "canaries": canary_summaries,
        "models": model_summaries,
        "weak_seats": weak_seats,
        "recommended_action": action,
    }
    if save:
        save_report(report, snapshot_dir)
    return report
