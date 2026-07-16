"""Brainstorm mode: independent divergence, cross-pollination, then curation."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from porin import stream_event
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from quorate import runlog
from quorate.api import query_judge, query_model, quorum_health
from quorate.config import (
    Message,
    ModelCallResult,
    ModelEntry,
    ReasoningEffort,
    api_keys,
    brainstorm_models,
    is_error,
    resolved_judge,
    resolved_judge_fallback,
)

BRAINSTORM_LENSES: tuple[tuple[str, str], ...] = (
    (
        "First principles",
        "Remove inherited assumptions and rebuild the opportunity from fundamental needs.",
    ),
    (
        "Unmet user tension",
        "Start from an awkward, expensive, or emotionally charged user problem.",
    ),
    (
        "Contrarian",
        "Invert the conventional answer and identify when the inversion becomes valuable.",
    ),
    (
        "Adjacent industry",
        "Import a proven mechanism from a structurally similar but culturally distant field.",
    ),
    (
        "Constraint removal",
        "Imagine one major constraint disappears, then find the newly possible design.",
    ),
    (
        "Recombination",
        "Combine two existing capabilities whose interaction creates a new behavior.",
    ),
    (
        "Long horizon",
        "Work backward from a plausible five-year change in technology or behavior.",
    ),
    (
        "Cheap experiment",
        "Seek an idea whose core value can be tested quickly with minimal irreversible work.",
    ),
)

GENERATOR_SYSTEM = """You are one generator in a divergent brainstorming exercise.

Your assigned lens is {lens}: {instruction}

Generate exactly four distinct ideas through this lens. Do not rank, critique, or converge. Each
idea needs a short name, a concrete mechanism, and one sentence explaining why it is non-obvious.
Avoid generic feature lists and keep the response under 500 words. You have not seen other models.
"""

CROSS_POLLINATION_SYSTEM = """You are in the cross-pollination round of a brainstorming exercise.

Your assigned lens remains {lens}: {instruction}

Create exactly two new ideas by mutating, combining, or productively conflicting your seed ideas
with one peer's seed ideas. Do not repeat, rank, or merely summarize either source. Give each idea
a short name, its mechanism, and the new connection that produced it. Keep the response under 300
words.
"""

CURATOR_SYSTEM = """You are a creative curator, not a consensus judge. The material contains seed
ideas and one cross-pollination round from several model families. Cluster genuine duplicates, but
do not reward an idea merely because several models repeated it.

Return exactly these sections:

## Shortlist
Six distinct ideas. For each, give a short name, a two-sentence mechanism, the first cheap test,
and four scores from 1 to 5 for novelty, leverage, feasibility, and distinctness.

## Wildcard
Preserve one high-novelty idea that is too uncertain for the shortlist and explain what would make
it unexpectedly important.

## Pattern map
Name the two or three opportunity territories that connect the strongest ideas and the key tension
between them.

Prefer specific mechanisms over polished prose. Preserve productive disagreement and do not invent
ideas absent from the supplied material.
"""

CURATOR_HEADINGS = ("## shortlist", "## wildcard", "## pattern map")


def _valid_curation(response: str) -> bool:
    """Require the curator's complete user-facing contract."""
    normalized = response.lower()
    return not is_error(response) and all(heading in normalized for heading in CURATOR_HEADINGS)


async def _run_prompt_round(
    entries: list[ModelEntry],
    prompts: list[list[Message]],
    *,
    timeout: float,
    max_tokens: int,
) -> list[ModelCallResult]:
    """Run model-specific prompts concurrently through the standard provider router."""
    keys = api_keys()
    async with httpx.AsyncClient() as client:
        tasks = [
            query_model(
                client,
                keys,
                entry,
                messages,
                max_tokens=max_tokens,
                timeout=timeout,
                effort=ReasoningEffort.MEDIUM,
            )
            for entry, messages in zip(entries, prompts, strict=True)
        ]
        rows = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[ModelCallResult] = []
    for entry, row in zip(entries, rows, strict=True):
        if isinstance(row, BaseException):
            results.append(
                ModelCallResult(
                    name=entry.name,
                    model_id=entry.model.rsplit("/", 1)[-1],
                    response=f"[Error: {entry.name} failed]",
                    provider="exception",
                    diagnostics=("orchestrator:exception",),
                )
            )
        else:
            results.append(row)
    return results


def _result_with_metadata(
    result: ModelCallResult,
    *,
    lens: str,
    source_model: str | None = None,
) -> dict[str, Any]:
    row = result.to_dict()
    row["lens"] = lens
    if source_model:
        row["source_model"] = source_model
    return row


async def run_brainstorm(
    question: str,
    context: str | None = None,
    models: list[ModelEntry] | None = None,
    timeout: float = 180,
    console: Console | None = None,
    json_output: bool = False,
) -> str | dict[str, Any]:
    """Generate independently, cross-pollinate once, and curate a shortlist."""
    console = console or Console()
    models = models or brainstorm_models()
    full_question = f"{context}\n\n{question}" if context else question
    start = time.monotonic()
    lenses = [BRAINSTORM_LENSES[index % len(BRAINSTORM_LENSES)] for index in range(len(models))]

    console.print("\n[bold cyan]INDEPENDENT GENERATION[/bold cyan]")
    seed_prompts = [
        [
            Message.system(GENERATOR_SYSTEM.format(lens=lens, instruction=instruction)),
            Message.user(full_question),
        ]
        for lens, instruction in lenses
    ]
    seed_results = await _run_prompt_round(
        models,
        seed_prompts,
        timeout=timeout,
        max_tokens=1400,
    )
    seed_json: list[dict[str, Any]] = []
    for result, (lens, _instruction) in zip(seed_results, lenses, strict=True):
        row = _result_with_metadata(result, lens=lens)
        seed_json.append(row)
        if not result.is_error:
            console.print(Panel(Markdown(result.response), title=f"{result.name} · {lens}"))
        else:
            console.print(f"[red]{result.name}: {result.response}[/red]")
        if json_output:
            stream_event("generation", row)

    success_count, quorum_target, quorum_achieved = quorum_health(seed_results)
    if not quorum_achieved:
        result: dict[str, Any] = {
            "question": question,
            "phases": {"generation": seed_json},
            "success_count": success_count,
            "failed_count": len(seed_results) - success_count,
            "quorum_target": quorum_target,
            "quorum_achieved": False,
            "duration_s": round(time.monotonic() - start, 1),
        }
        return result if json_output else ""

    successful = [
        (entry, result, lens)
        for entry, result, lens in zip(models, seed_results, lenses, strict=True)
        if not result.is_error
    ]
    console.print("\n[bold cyan]CROSS-POLLINATION[/bold cyan]")
    cross_entries: list[ModelEntry] = []
    cross_prompts: list[list[Message]] = []
    cross_metadata: list[tuple[str, str]] = []
    for index, (entry, seed, (lens, instruction)) in enumerate(successful):
        peer_entry, peer_seed, _peer_lens = successful[(index + 1) % len(successful)]
        cross_entries.append(entry)
        cross_metadata.append((lens, peer_entry.name))
        cross_prompts.append(
            [
                Message.system(CROSS_POLLINATION_SYSTEM.format(lens=lens, instruction=instruction)),
                Message.user(
                    f"Original opportunity:\n{full_question}\n\n"
                    f"Your seed ideas:\n{seed.response}\n\n"
                    f"Peer seed ideas from {peer_entry.name}:\n{peer_seed.response}"
                ),
            ]
        )

    cross_results = await _run_prompt_round(
        cross_entries,
        cross_prompts,
        timeout=timeout,
        max_tokens=1000,
    )
    cross_json: list[dict[str, Any]] = []
    for result, (lens, source_model) in zip(cross_results, cross_metadata, strict=True):
        row = _result_with_metadata(result, lens=lens, source_model=source_model)
        cross_json.append(row)
        if not result.is_error:
            console.print(Panel(Markdown(result.response), title=f"{result.name} · hybrid"))
        else:
            console.print(f"[red]{result.name}: {result.response}[/red]")
        if json_output:
            stream_event("cross_pollination", row)

    traces = []
    for entry, seed, (lens, _instruction) in successful:
        traces.append(f"### {entry.name} · {lens} · seed\n{seed.response}")
    for result, (lens, source_model) in zip(cross_results, cross_metadata, strict=True):
        if not result.is_error:
            traces.append(
                f"### {result.name} · {lens} · hybrid with {source_model}\n{result.response}"
            )

    console.print("\n[bold cyan]CURATED SHORTLIST[/bold cyan]")
    curator = resolved_judge()
    fallback = resolved_judge_fallback()
    curator_messages = [
        Message.system(CURATOR_SYSTEM),
        Message.user(f"Opportunity: {full_question}\n\n" + "\n\n".join(traces)),
    ]
    curated = await query_judge(
        curator,
        curator_messages,
        max_tokens=4096,
        timeout=300,
        effort=ReasoningEffort.HIGH,
    )
    curator_used = curator
    if not _valid_curation(curated) and fallback != curator:
        curated = await query_judge(
            fallback,
            curator_messages,
            max_tokens=4096,
            timeout=300,
            effort=ReasoningEffort.HIGH,
        )
        curator_used = fallback
    if not _valid_curation(curated):
        curated = "[Error: Curator response did not include the required sections]"
        console.print(f"[red]{curated}[/red]")
        if json_output:
            return {
                "question": question,
                "error": "Brainstorm curation failed its output contract",
                "phases": {
                    "generation": seed_json,
                    "cross_pollination": cross_json,
                    "curation": {"model": curator_used, "error": curated},
                },
                "success_count": success_count,
                "failed_count": len(seed_results) - success_count,
                "quorum_target": quorum_target,
                "quorum_achieved": True,
                "duration_s": round(time.monotonic() - start, 1),
            }
        return ""

    console.print(Panel(Markdown(curated), title="[bold green]Curator[/bold green]"))
    curator_data: dict[str, Any] = {"model": curator_used, "response": curated}
    if curator_used != curator:
        curator_data["preferred_model"] = curator
    if json_output:
        stream_event("curation", curator_data)

    duration = time.monotonic() - start
    record = runlog.build_record(
        mode="brainstorm",
        results=seed_results,
        extra_results=cross_results,
        total_duration_s=duration,
        judge_model=curator_used,
    )
    runlog.append(record)
    footer_lines, summary = runlog.format_footer(
        seed_results,
        duration,
        extra_results=cross_results,
    )
    console.print()
    for line in footer_lines:
        console.print(f"[dim]{line}[/dim]")
    console.print(f"[dim]{summary}[/dim]")

    if json_output:
        return {
            "question": question,
            "phases": {
                "generation": seed_json,
                "cross_pollination": cross_json,
                "curation": curator_data,
            },
            "success_count": success_count,
            "failed_count": len(seed_results) - success_count,
            "quorum_target": quorum_target,
            "quorum_achieved": True,
            "duration_s": round(duration, 1),
        }
    return curated
