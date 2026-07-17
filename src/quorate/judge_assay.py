"""Paired, deterministic judge-role assay for Fable and GPT."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from quorate.api import run_parallel
from quorate.config import Message, ModelCallResult, ModelEntry, ReasoningEffort

SUITE_VERSION = "2026-07-17"
FABLE = ModelEntry("Claude-Fable-5", "anthropic/claude-fable-5")
GPT = ModelEntry("GPT-5.6-Sol", "openai/gpt-5.6-sol")
MODELS = (FABLE, GPT)
FINAL_MARKER = re.compile(r"FINAL_DECISION\s*:\s*([AB])\b", re.IGNORECASE)
SYSTEM_PROMPT = """You are the final judge of a synthetic decision case.
Use only the stated evidence. Choose recommendation A or B; do not invent missing facts.
Explain the decisive calculation or constraint briefly, then end with exactly one line in this form:
FINAL_DECISION: A
Replace A with B when appropriate. Do not emit any other FINAL_DECISION marker."""


@dataclass(frozen=True)
class Case:
    id: str
    domain: str
    prompt: str
    expected: str


CASES: tuple[Case, ...] = (
    Case(
        "capacity-stability",
        "capacity",
        "A queue receives 12 jobs per minute. Recommendation A uses a worker pool that "
        "sustainably completes 15 jobs per minute. Recommendation B uses a cheaper pool that "
        "sustainably completes 10 jobs per minute. The objective is stable operation without "
        "unbounded backlog. Choose A or B.",
        "A",
    ),
    Case(
        "privacy-minimization",
        "privacy",
        "A report needs only daily totals. Recommendation A uploads identifiable event rows and "
        "retains them indefinitely. Recommendation B aggregates locally, uploads only totals, and "
        "deletes source rows after the approved retention period. Accuracy is identical. Choose A "
        "or B.",
        "B",
    ),
    Case(
        "expected-value",
        "expected-value",
        "Recommendation A yields 100 units with probability 0.70 and loses 80 units otherwise. "
        "Recommendation B yields 60 units with probability 0.90 and loses 10 units otherwise. "
        "Choose the higher expected value.",
        "B",
    ),
    Case(
        "parallel-reliability",
        "reliability",
        "Recommendation A uses two independent components, each 95% reliable, and succeeds when "
        "either component works. Recommendation B uses one component that is 98% reliable. Choose "
        "the design with higher service reliability.",
        "A",
    ),
    Case(
        "contract-expected-cost",
        "contracts",
        "Recommendation A is a fixed contract costing 100 units plus a 15% chance of a 40-unit "
        "overrun. Recommendation B costs 108 units with no overrun. All other terms are equal. "
        "Choose the lower expected cost.",
        "A",
    ),
    Case(
        "base-rate-confirmation",
        "statistics",
        "A defect affects 1% of items. A test has 90% sensitivity and 90% specificity. "
        "An item tests positive. Recommendation A treats the item as more likely defective "
        "than sound. "
        "Recommendation B requires confirmation because false positives dominate at this base "
        "rate. Choose A or B.",
        "B",
    ),
    Case(
        "credential-incident",
        "operational-risk",
        "A live credential appears in a public log. Recommendation A immediately revokes and "
        "rotates it, then removes the log and investigates use. Recommendation B deletes the log "
        "but keeps the credential to avoid disruption. Choose A or B.",
        "A",
    ),
    Case(
        "least-privilege",
        "access-control",
        "Recommendation A gives a team one shared administrator account for convenience. "
        "Recommendation B gives named accounts only the permissions each role needs and logs each "
        "action. Delivery speed is otherwise equal. Choose A or B.",
        "B",
    ),
    Case(
        "reversible-rollout",
        "operational-risk",
        "A release has uncertain failure risk. Recommendation A deploys to a 5% canary with an "
        "automatic rollback threshold before wider release. Recommendation B deploys globally with "
        "no rollback because rollback preparation adds one hour. Choose A or B.",
        "A",
    ),
    Case(
        "critical-path",
        "scheduling",
        "Recommendation A performs two dependent tasks taking 4 and 5 hours, for 9 hours total. "
        "Recommendation B performs independent 6-hour and 4-hour tasks in parallel, followed by a "
        "1-hour integration, for 7 hours total. Choose the shorter plan.",
        "B",
    ),
    Case(
        "sla-threshold",
        "reliability",
        "The service may be unavailable for at most 30 minutes in a 30-day month. Recommendation A "
        "offers 99.9% availability. Recommendation B offers 99.95% availability. Choose the option "
        "that meets the stated threshold.",
        "B",
    ),
    Case(
        "insurance-expected-cost",
        "expected-value",
        "An uninsured loss of 5,000 units has a 2% annual probability. Recommendation A "
        "accepts the risk. Recommendation B buys insurance for 130 units with no other benefit. "
        "Choose the lower "
        "expected annual cost.",
        "A",
    ),
    Case(
        "selection-proxy",
        "privacy",
        "Recommendation A ranks applicants using a postcode proxy that closely reconstructs a "
        "protected attribute, although that attribute is irrelevant to job performance. "
        "Recommendation B removes the proxy and validates job-related features on a held-out "
        "sample. Choose A or B.",
        "B",
    ),
    Case(
        "irreversible-payment-control",
        "operational-risk",
        "Recommendation A requires two independent approvals for an irreversible high-value "
        "payment. Recommendation B lets one operator create and approve it to save five minutes. "
        "The control "
        "objective is preventing unauthorized irreversible transfers. Choose A or B.",
        "A",
    ),
    Case(
        "power-comparison",
        "statistics",
        "Two unbiased experiments have the same effect size, variance, and cost per observation. "
        "Recommendation A uses 20 observations. Recommendation B uses 200 observations. "
        "The objective is greater statistical power. Choose A or B.",
        "B",
    ),
    Case(
        "restore-testing",
        "reliability",
        "Both recommendations take daily encrypted backups. Recommendation A also performs and "
        "records a full restore test every month. Recommendation B never tests restoration because "
        "the backup job reports success. Choose the stronger recovery design.",
        "A",
    ),
    Case(
        "outcome-metric",
        "measurement",
        "The objective is fewer customer billing errors. Recommendation A rewards the number of "
        "cases processed. Recommendation B measures independently verified billing errors after "
        "processing. "
        "Choose the metric aligned with the objective.",
        "B",
    ),
    Case(
        "contract-notice",
        "contracts",
        "A vendor breached its service level. The contract permits termination after 30 days' "
        "written notice. Recommendation A issues notice now and terminates after 30 days. "
        "Recommendation B terminates immediately despite no emergency or immediate-termination "
        "clause. Choose A or B.",
        "A",
    ),
    Case(
        "multiple-comparisons",
        "statistics",
        "A team tests 20 unrelated hypotheses at a 5% threshold. Recommendation A reports the "
        "smallest uncorrected p-value as confirmed. Recommendation B applies a predeclared "
        "multiple-testing "
        "correction before claiming a finding. Choose A or B.",
        "B",
    ),
    Case(
        "uncertain-forecast",
        "expected-value",
        "A demand forecast is highly uncertain and both strategies have similar modeled value. "
        "Recommendation A runs a small reversible pilot that resolves the key uncertainty before "
        "committing. Recommendation B makes the full irreversible investment immediately. "
        "Choose the option with lower avoidable decision risk.",
        "A",
    ),
)


def parse_decision(response: str) -> str | None:
    markers = FINAL_MARKER.findall(response)
    return markers[0].upper() if len(markers) == 1 else None


def attempt_row(case: Case, result: ModelCallResult) -> dict[str, Any]:
    reachable = not result.is_error
    decision = parse_decision(result.response) if reachable else None
    contract_completed = decision is not None
    return {
        "case": case.id,
        "domain": case.domain,
        "expected": case.expected,
        "model": result.name,
        "model_id": result.model_id,
        "provider": result.provider,
        "reachable": reachable,
        "contract_completed": contract_completed,
        "decision": decision,
        "correct": contract_completed and decision == case.expected,
        "latency_s": round(result.latency_s, 2),
        "response_chars": len(result.response) if reachable else 0,
        "diagnostics": list(result.diagnostics),
    }


def summarize_model(name: str, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in attempts if row["model"] == name]
    latencies = [float(row["latency_s"]) for row in rows if row["reachable"]]
    providers = Counter(str(row["provider"]) for row in rows)
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "model": name,
        "cases": len(rows),
        "reachable": sum(bool(row["reachable"]) for row in rows),
        "contract_completed": sum(bool(row["contract_completed"]) for row in rows),
        "correct": correct,
        "usable_accuracy": round(correct / len(rows), 3) if rows else 0.0,
        "mean_latency_s": round(mean(latencies), 2) if latencies else None,
        "median_latency_s": round(median(latencies), 2) if latencies else None,
        "providers": dict(sorted(providers.items())),
    }


def routing_decision(summaries: list[dict[str, Any]]) -> dict[str, str]:
    by_name = {str(row["model"]): row for row in summaries}
    comparison_floor = 18
    for model in MODELS:
        summary = by_name[model.name]
        if (
            int(summary["reachable"]) < comparison_floor
            or int(summary["contract_completed"]) < comparison_floor
        ):
            return {
                "action": "invalid-run",
                "reason": (
                    "At least one route produced fewer than 18 reachable, contract-compliant "
                    "results; do not infer relative capability from a provider outage."
                ),
            }
    fable_correct = int(by_name[FABLE.name]["correct"])
    gpt_correct = int(by_name[GPT.name]["correct"])
    gpt_contracts = int(by_name[GPT.name]["contract_completed"])
    if gpt_correct - fable_correct >= 2 and gpt_contracts >= 19:
        return {
            "action": "promote-gpt",
            "reason": (
                "GPT led by at least two usable correct cases and completed at least 19 contracts."
            ),
        }
    return {
        "action": "retain-fable",
        "reason": (
            "GPT did not clear the predeclared two-case quality margin; retain the independent "
            "incumbent judge."
        ),
    }


async def evaluate_case(case: Case, semaphore: asyncio.Semaphore) -> list[dict[str, Any]]:
    async with semaphore:
        results = await run_parallel(
            list(MODELS),
            [Message.system(SYSTEM_PROMPT), Message.user(case.prompt)],
            max_tokens=1024,
            timeout=300,
            effort=ReasoningEffort.HIGH,
        )
    return [attempt_row(case, result) for result in results]


async def run_assay(concurrency: int = 4) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    nested = await asyncio.gather(*(evaluate_case(case, semaphore) for case in CASES))
    attempts = [row for rows in nested for row in rows]
    summaries = [summarize_model(model.name, attempts) for model in MODELS]
    paired = {
        "fable_only_correct": sum(
            1
            for case in CASES
            if next(
                row for row in attempts if row["case"] == case.id and row["model"] == FABLE.name
            )["correct"]
            and not next(
                row for row in attempts if row["case"] == case.id and row["model"] == GPT.name
            )["correct"]
        ),
        "gpt_only_correct": sum(
            1
            for case in CASES
            if next(row for row in attempts if row["case"] == case.id and row["model"] == GPT.name)[
                "correct"
            ]
            and not next(
                row for row in attempts if row["case"] == case.id and row["model"] == FABLE.name
            )["correct"]
        ),
    }
    paired["same_outcome"] = len(CASES) - paired["fable_only_correct"] - paired["gpt_only_correct"]
    return {
        "schema_version": 1,
        "suite_version": SUITE_VERSION,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "models": [entry.model for entry in MODELS],
        "cases": len(CASES),
        "concurrency": concurrency,
        "effort": ReasoningEffort.HIGH.value,
        "output_contract": "Exactly one FINAL_DECISION marker with A or B",
        "promotion_rule": (
            "Promote GPT only if it leads Fable by at least two usable correct cases and completes "
            "at least 19 of 20 output contracts."
        ),
        "summaries": summaries,
        "paired": paired,
        "decision": routing_decision(summaries),
        "attempts": attempts,
        "responses_retained": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    report = asyncio.run(run_assay(arguments.concurrency))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(arguments.output), **report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
