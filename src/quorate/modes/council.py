"""Council mode: blind phase, debate rounds, judge synthesis, critique."""

from __future__ import annotations

import re
import time
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from quorate import runlog
from quorate.api import query_judge, run_parallel
from quorate.config import (
    Message,
    ModelEntry,
    ReasoningEffort,
    is_error,
    resolved_council,
    resolved_critique,
    resolved_judge,
)
from porin import stream_event
from quorate.prompts import (
    BLIND_SYSTEM,
    CHALLENGER_ADDITION,
    CRITIQUE_SYSTEM,
    judge_system,
    debate_system,
)

CONFIDENCE_RE = re.compile(r"(?i)\*{0,2}Confidence\*{0,2}:?\s*(\d{1,2})\s*(?:/\s*10)")


def _sanitize(content: str) -> str:
    """Sanitize speaker content to prevent prompt injection."""
    return (
        content
        .replace("SYSTEM:", "[SYSTEM]:")
        .replace("INSTRUCTION:", "[INSTRUCTION]:")
        .replace("IGNORE PREVIOUS", "[IGNORE PREVIOUS]")
        .replace("OVERRIDE:", "[OVERRIDE]:")
    )


async def run_council(
    question: str,
    context: str | None = None,
    models: list[ModelEntry] | None = None,
    rounds: int = 1,
    timeout: float = 120,
    effort: ReasoningEffort | None = None,
    judge_model: str | None = None,
    critic_model: str | None = None,
    no_critic: bool = False,
    console: Console | None = None,
    json_output: bool = False,
) -> str | dict[str, Any]:
    """Run full council deliberation: blind → debate → judge → critique.

    Returns judge_response (str) normally, or structured dict when json_output=True.
    """
    import sys
    if json_output:
        console = console or Console(file=sys.stderr)
    else:
        console = console or Console()
    models = models or resolved_council()
    judge = resolved_judge(judge_model)
    critique = resolved_critique(critic_model)

    full_question = f"{context}\n\n{question}" if context else question
    start = time.monotonic()

    # JSON accumulator
    result: dict[str, Any] = {"question": question, "phases": {}} if json_output else {}

    # --- BLIND PHASE ---
    console.print("\n[bold cyan]BLIND PHASE[/bold cyan]")
    blind_messages = [Message.system(BLIND_SYSTEM), Message.user(full_question)]

    blind_results = await run_parallel(
        models, blind_messages, max_tokens=1024, timeout=timeout, effort=effort
    )

    blind_claims: dict[str, str] = {}
    blind_json: list[dict[str, Any]] = []
    for mcr in blind_results:
        if not mcr.is_error:
            blind_claims[mcr.name] = mcr.response
            blind_json.append(mcr.to_dict())
            console.print(Panel(
                Markdown(mcr.response), title=f"[bold]{mcr.name}[/bold]", border_style="dim"
            ))
        else:
            blind_json.append(mcr.to_dict())
            console.print(f"[red]{mcr.name}: {mcr.response}[/red]")
        if json_output:
            stream_event("blind", mcr.to_dict())

    blind_failed = [mcr for mcr in blind_results if mcr.is_error]
    if blind_failed:
        names = ", ".join(mcr.name for mcr in blind_failed)
        console.print(f"\n[bold red]⚠ {len(blind_failed)}/{len(blind_results)} models failed: {names}[/bold red]")

    if json_output:
        result["phases"]["blind"] = blind_json
        result["failed_count"] = len(blind_failed)

    if len(blind_claims) < 2:
        console.print("[red]Too few models responded in blind phase.[/red]")
        if json_output:
            result["error"] = "Too few models responded in blind phase"
            return result
        return ""

    # --- DEBATE ROUNDS ---
    blind_summary = "\n\n".join(
        f"**{name}** (blind claim): {claim}" for name, claim in blind_claims.items()
    )
    conversation: list[tuple[str, str]] = []
    debate_json: list[dict[str, Any]] = []
    for round_num in range(1, rounds + 1):
        console.print(f"\n[bold cyan]DEBATE ROUND {round_num}[/bold cyan]")
        challenger_idx = (round_num - 1) % len(models)

        # Build debate context from prior conversation (blind_summary set above)
        conv_summary = "\n\n".join(
            f"**{name}**: {_sanitize(text)}"
            for name, text in conversation
        )

        previous_speakers = ""
        for idx, entry in enumerate(models):
            if entry.name not in blind_claims:
                continue

            speaker_prompt = debate_system(entry.name, round_num, previous_speakers or "None yet")
            is_challenger = idx == challenger_idx
            if is_challenger:
                speaker_prompt += CHALLENGER_ADDITION

            context_block = f"BLIND CLAIMS:\n{blind_summary}"
            if conv_summary:
                context_block += f"\n\nPRIOR CONVERSATION:\n{conv_summary}"

            msgs = [
                Message.system(speaker_prompt),
                Message.user(f"{context_block}\n\nOriginal question: {full_question}"),
            ]

            # Query this speaker sequentially (debate is sequential)
            from quorate.api import query_model
            from quorate.config import api_keys
            import httpx

            keys = api_keys()
            async with httpx.AsyncClient() as client:
                mcr = await query_model(
                    client, keys, entry, msgs, max_tokens=2048, timeout=timeout, effort=effort
                )

            entry_dict = mcr.to_dict()
            entry_dict["round"] = round_num
            entry_dict["role"] = "challenger" if is_challenger else "speaker"

            if not mcr.is_error:
                conversation.append((mcr.name, mcr.response))
                debate_json.append(entry_dict)
                role = " [yellow](challenger)[/yellow]" if is_challenger else ""
                console.print(Panel(
                    Markdown(mcr.response), title=f"[bold]{mcr.name}[/bold]{role}", border_style="dim"
                ))
                previous_speakers = (
                    f"{previous_speakers}, {mcr.name}" if previous_speakers else mcr.name
                )
            else:
                debate_json.append(entry_dict)
                console.print(f"[red]{mcr.name}: {mcr.response}[/red]")
            if json_output:
                stream_event("debate", entry_dict)

    if json_output:
        result["phases"]["debate"] = debate_json

    # --- JUDGE ---
    console.print("\n[bold cyan]JUDGE SYNTHESIS[/bold cyan]")
    all_text = blind_summary
    if conversation:
        debate_text = "\n\n".join(
            f"**{name}**: {text}" for name, text in conversation
        )
        all_text += f"\n\n--- DEBATE ---\n\n{debate_text}"

    failed_names = [mcr.name for mcr in blind_failed] if blind_failed else None
    judge_prompt = judge_system(len(models), failed_names)
    judge_messages = [
        Message.system(judge_prompt),
        Message.user(f"Question: {full_question}\n\n{all_text}"),
    ]
    judge_response = await query_judge(
        judge, judge_messages, max_tokens=16384, timeout=300, effort=ReasoningEffort.HIGH
    )

    if is_error(judge_response):
        console.print(f"[red]Judge failed: {judge_response}[/red]")
        if json_output:
            result["phases"]["judge"] = {"model": judge, "error": judge_response}
            return result
        return ""

    console.print(Panel(Markdown(judge_response), title="[bold green]Judge[/bold green]", border_style="green"))
    judge_data = {"model": judge, "response": judge_response}
    if json_output:
        result["phases"]["judge"] = judge_data
        stream_event("judge", judge_data)

    # --- CRITIQUE ---
    if not no_critic:
        console.print("\n[bold cyan]CRITIQUE[/bold cyan]")
        critique_messages = [
            Message.system(CRITIQUE_SYSTEM),
            Message.user(f"Question: {full_question}\n\nJudge synthesis:\n{judge_response}"),
        ]
        critique_response = await query_judge(
            critique, critique_messages, max_tokens=4096, timeout=120
        )
        if not is_error(critique_response):
            console.print(Panel(
                Markdown(critique_response),
                title="[bold yellow]Critique[/bold yellow]",
                border_style="yellow",
            ))
            critique_data = {"model": str(critique), "response": critique_response}
            if json_output:
                result["phases"]["critique"] = critique_data
                stream_event("critique", critique_data)

    duration = time.monotonic() - start
    if blind_failed:
        names = ", ".join(mcr.name for mcr in blind_failed)
        console.print(f"\n[bold red]⚠ Partial council: {len(blind_failed)}/{len(blind_results)} models failed ({names})[/bold red]")
    record = runlog.build_record(
        mode="council",
        results=blind_results,
        total_duration_s=duration,
        judge_model=judge,
    )
    runlog.append(record)
    footer_lines, summary = runlog.format_footer(blind_results, duration)
    console.print()
    for line in footer_lines:
        console.print(f"[dim]{line}[/dim]")
    console.print(f"[dim]{summary}[/dim]")

    if json_output:
        result["duration_s"] = round(duration, 1)
        return result

    return judge_response
