"""Council mode: blind phase, debate rounds, judge synthesis, critique."""

from __future__ import annotations

import re
import time
from typing import Any

from porin import stream_event
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from quorate import runlog
from quorate.api import query_judge, quorum_health, run_parallel
from quorate.config import (
    Message,
    ModelEntry,
    ReasoningEffort,
    is_error,
    resolved_council,
    resolved_critique,
    resolved_critique_fallback,
    resolved_judge,
    resolved_judge_fallback,
)
from quorate.heavyskill import prune_cot, shuffle_traces
from quorate.prompts import (
    BLIND_SYSTEM,
    CHALLENGER_ADDITION,
    CRITIQUE_SYSTEM,
    debate_system,
    judge_system,
)

CONFIDENCE_RE = re.compile(r"(?i)\*{0,2}Confidence\*{0,2}:?\s*(\d{1,2})\s*(?:/\s*10)")


def _sanitize(content: str) -> str:
    """Sanitize speaker content to prevent prompt injection."""
    return (
        content.replace("SYSTEM:", "[SYSTEM]:")
        .replace("INSTRUCTION:", "[INSTRUCTION]:")
        .replace("IGNORE PREVIOUS", "[IGNORE PREVIOUS]")
        .replace("OVERRIDE:", "[OVERRIDE]:")
    )


def _build_synthesis_text(
    blind_claims: dict[str, str],
    conversation: list[tuple[str, str]],
    *,
    shuffle_enabled: bool,
    prune_enabled: bool,
) -> tuple[str, list[str], int | None]:
    if not shuffle_enabled and not prune_enabled:
        all_text = "\n\n".join(
            f"**{name}** (blind claim): {claim}" for name, claim in blind_claims.items()
        )
        if conversation:
            debate_text = "\n\n".join(f"**{name}**: {text}" for name, text in conversation)
            all_text += f"\n\n--- DEBATE ---\n\n{debate_text}"
        return all_text, [], None

    traces: list[dict[str, str]] = []
    # HeavySkill (Wang et al., ICML 2026, arXiv:2605.02396)
    for name, claim in blind_claims.items():
        traces.append(
            {
                "speaker": name,
                "label": "blind claim",
                "content": prune_cot(claim) if prune_enabled else claim,
            }
        )
    for name, text in conversation:
        traces.append(
            {
                "speaker": name,
                "label": "debate",
                "content": prune_cot(text) if prune_enabled else text,
            }
        )

    shuffle_seed: int | None = None
    if shuffle_enabled:
        # HeavySkill (Wang et al., ICML 2026, arXiv:2605.02396)
        shuffle_seed = time.time_ns()
        traces = shuffle_traces(traces, seed=shuffle_seed)

    rendered = "\n\n".join(
        f"**{trace['speaker']}** ({trace['label']}): {trace['content']}" for trace in traces
    )
    order = [f"{trace['speaker']} [{trace['label']}]" for trace in traces]
    return rendered, order, shuffle_seed


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
    shuffle_traces_enabled: bool = False,
    prune_cot_enabled: bool = False,
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
    judge_fallback = resolved_judge_fallback()
    critique = resolved_critique(critic_model)
    critique_fallback = resolved_critique_fallback()

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
            console.print(
                Panel(Markdown(mcr.response), title=f"[bold]{mcr.name}[/bold]", border_style="dim")
            )
        else:
            blind_json.append(mcr.to_dict())
            console.print(f"[red]{mcr.name}: {mcr.response}[/red]")
        if json_output:
            stream_event("blind", mcr.to_dict())

    blind_failed = [mcr for mcr in blind_results if mcr.is_error]
    success_count, quorum_target, quorum_achieved = quorum_health(blind_results)
    if blind_failed:
        names = ", ".join(mcr.name for mcr in blind_failed)
        console.print(
            f"\n[bold red]⚠ {len(blind_failed)}/{len(blind_results)} models failed: {names}[/bold"
            " red]"
        )

    if json_output:
        result["phases"]["blind"] = blind_json
        result["success_count"] = success_count
        result["failed_count"] = len(blind_failed)
        result["quorum_target"] = quorum_target
        result["quorum_achieved"] = quorum_achieved

    if not quorum_achieved:
        message = (
            f"No quorum in blind phase: {success_count} successful responses; "
            f"{quorum_target} required"
        )
        console.print(f"[red]{message}.[/red]")
        if json_output:
            result["error"] = message
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
        conv_summary = "\n\n".join(f"**{name}**: {_sanitize(text)}" for name, text in conversation)

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
            import httpx

            from quorate.api import query_model
            from quorate.config import api_keys

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
                console.print(
                    Panel(
                        Markdown(mcr.response),
                        title=f"[bold]{mcr.name}[/bold]{role}",
                        border_style="dim",
                    )
                )
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
    all_text, trace_order, shuffle_seed = _build_synthesis_text(
        blind_claims,
        conversation,
        shuffle_enabled=shuffle_traces_enabled,
        prune_enabled=prune_cot_enabled,
    )
    if shuffle_traces_enabled and trace_order:
        console.print(
            f"[dim]Shuffled synthesis trace order (seed={shuffle_seed}):"
            f" {' -> '.join(trace_order)}[/dim]"
        )

    failed_names = [mcr.name for mcr in blind_failed] if blind_failed else None
    judge_prompt = judge_system(len(models), failed_names)
    judge_messages = [
        Message.system(judge_prompt),
        Message.user(f"Question: {full_question}\n\n{all_text}"),
    ]
    judge_response = await query_judge(
        judge, judge_messages, max_tokens=16384, timeout=300, effort=ReasoningEffort.HIGH
    )

    judge_used = judge
    if is_error(judge_response) and judge_fallback != judge:
        console.print(
            "[yellow]Preferred judge failed; trying subscription fallback "
            f"{judge_fallback}.[/yellow]"
        )
        judge_response = await query_judge(
            judge_fallback,
            judge_messages,
            max_tokens=16384,
            timeout=300,
            effort=ReasoningEffort.HIGH,
        )
        judge_used = judge_fallback

    if is_error(judge_response):
        console.print(f"[red]Judge failed: {judge_response}[/red]")
        if json_output:
            result["phases"]["judge"] = {"model": judge_used, "error": judge_response}
            return result
        return ""

    console.print(
        Panel(
            Markdown(judge_response), title="[bold green]Judge[/bold green]", border_style="green"
        )
    )
    judge_data = {
        "model": judge_used,
        "response": judge_response,
        "shuffle_traces": shuffle_traces_enabled,
        "prune_cot": prune_cot_enabled,
    }
    if judge_used != judge:
        judge_data["preferred_model"] = judge
    if shuffle_seed is not None:
        judge_data["shuffle_seed"] = shuffle_seed
        judge_data["trace_order"] = trace_order
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
        critique_used = critique
        if is_error(critique_response) and critique_fallback != critique:
            console.print(
                "[yellow]Preferred critic failed; trying subscription fallback "
                f"{critique_fallback}.[/yellow]"
            )
            critique_response = await query_judge(
                critique_fallback,
                critique_messages,
                max_tokens=4096,
                timeout=120,
            )
            critique_used = critique_fallback
        if not is_error(critique_response):
            console.print(
                Panel(
                    Markdown(critique_response),
                    title="[bold yellow]Critique[/bold yellow]",
                    border_style="yellow",
                )
            )
            critique_data = {"model": str(critique_used), "response": critique_response}
            if critique_used != critique:
                critique_data["preferred_model"] = critique
            if json_output:
                result["phases"]["critique"] = critique_data
                stream_event("critique", critique_data)

    duration = time.monotonic() - start
    if blind_failed:
        names = ", ".join(mcr.name for mcr in blind_failed)
        console.print(
            f"\n[bold red]⚠ Partial council: {len(blind_failed)}/{len(blind_results)} models failed"
            f" ({names})[/bold red]"
        )
    outcome, outcome_note = runlog.prompt_outcome()
    record = runlog.build_record(
        mode="council",
        results=blind_results,
        total_duration_s=duration,
        judge_model=judge_used,
        outcome=outcome,
        outcome_note=outcome_note,
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
