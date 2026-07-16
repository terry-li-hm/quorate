"""Quick mode: parallel queries, no debate, no judge."""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from quorate import runlog
from quorate.api import quorum_health, run_parallel
from quorate.config import Message, ModelEntry, ReasoningEffort, quick_models


async def run_quick(
    question: str,
    context: str | None = None,
    models: list[ModelEntry] | None = None,
    timeout: float = 90,
    effort: ReasoningEffort | None = None,
    console: Console | None = None,
    json_output: bool = False,
) -> str | dict[str, Any]:
    """Run quick parallel query across all models."""
    console = console or Console()
    models = models or quick_models()
    full_question = f"{context}\n\n{question}" if context else question
    messages = [Message.user(full_question)]

    console.print(f"[dim](querying {len(models)} models in parallel...)[/dim]\n")
    start = time.monotonic()

    effective_effort = effort or ReasoningEffort.MEDIUM
    results = await run_parallel(
        models,
        messages,
        max_tokens=2048,
        timeout=max(timeout, 180),
        effort=effective_effort,
    )

    duration = time.monotonic() - start
    transcript_parts = []
    responses_json: list[dict[str, str]] = []

    for mcr in results:
        if mcr.is_error:
            console.print(f"[red]{mcr.name}: {mcr.response}[/red]")
        else:
            console.print(
                Panel(Markdown(mcr.response), title=f"[bold]{mcr.name}[/bold]", border_style="dim")
            )
            transcript_parts.append(f"### {mcr.name}\n{mcr.response}")
        responses_json.append(mcr.to_dict())

    failed = [mcr for mcr in results if mcr.is_error]
    success_count, quorum_target, quorum_achieved = quorum_health(results)
    if failed:
        names = ", ".join(mcr.name for mcr in failed)
        console.print(
            f"\n[bold red]⚠ {len(failed)}/{len(results)} models failed: {names}[/bold red]"
        )
    if not quorum_achieved:
        console.print(
            f"[bold red]No quorum: {success_count} successful responses; "
            f"{quorum_target} required.[/bold red]"
        )

    record = runlog.build_record(mode="quick", results=results, total_duration_s=duration)
    runlog.append(record)
    footer_lines, summary = runlog.format_footer(results, duration)
    console.print()
    for line in footer_lines:
        console.print(f"[dim]{line}[/dim]")
    console.print(f"[dim]{summary}[/dim]")

    if json_output:
        return {
            "question": question,
            "responses": responses_json,
            "success_count": success_count,
            "failed_count": len(failed),
            "quorum_target": quorum_target,
            "quorum_achieved": quorum_achieved,
            "duration_s": round(duration, 1),
        }

    return "\n\n".join(transcript_parts)
