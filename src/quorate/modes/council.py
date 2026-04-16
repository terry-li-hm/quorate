"""Council mode: blind phase, debate rounds, judge synthesis, critique."""

from __future__ import annotations

import re
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

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
from quorate.prompts import (
    BLIND_SYSTEM,
    CHALLENGER_ADDITION,
    CRITIQUE_SYSTEM,
    JUDGE_SYSTEM,
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
    domain: str | None = None,
    persona: str | None = None,
    console: Console | None = None,
) -> str:
    """Run full council deliberation: blind → debate → judge → critique."""
    console = console or Console()
    models = models or resolved_council()
    judge = resolved_judge(judge_model)
    critique = resolved_critique(critic_model)

    full_question = f"{context}\n\n{question}" if context else question
    start = time.monotonic()

    # --- BLIND PHASE ---
    console.print("\n[bold cyan]BLIND PHASE[/bold cyan]")
    blind_messages = [Message.system(BLIND_SYSTEM), Message.user(full_question)]

    blind_results = await run_parallel(
        models, blind_messages, max_tokens=1024, timeout=timeout, effort=effort
    )

    blind_claims: dict[str, str] = {}
    for name, _, response in blind_results:
        if not is_error(response):
            blind_claims[name] = response
            console.print(Panel(
                Markdown(response), title=f"[bold]{name}[/bold]", border_style="dim"
            ))
        else:
            console.print(f"[red]{name}: {response}[/red]")

    if len(blind_claims) < 2:
        console.print("[red]Too few models responded in blind phase.[/red]")
        return ""

    # --- DEBATE ROUNDS ---
    conversation: list[tuple[str, str]] = []
    for round_num in range(1, rounds + 1):
        console.print(f"\n[bold cyan]DEBATE ROUND {round_num}[/bold cyan]")
        challenger_idx = (round_num - 1) % len(models)

        # Build debate context from blind claims and prior conversation
        blind_summary = "\n\n".join(
            f"**{name}** (blind claim): {claim}" for name, claim in blind_claims.items()
        )
        conv_summary = "\n\n".join(
            f"**{name}**: {_sanitize(text)}"
            for name, text in conversation
        )

        previous_speakers = ""
        for idx, entry in enumerate(models):
            if entry.name not in blind_claims:
                continue

            speaker_prompt = debate_system(entry.name, round_num, previous_speakers or "None yet")
            if idx == challenger_idx:
                speaker_prompt += CHALLENGER_ADDITION

            context_block = f"BLIND CLAIMS:\n{blind_summary}"
            if conv_summary:
                context_block += f"\n\nPRIOR CONVERSATION:\n{conv_summary}"

            msgs = [
                Message.system(speaker_prompt),
                Message.user(f"{context_block}\n\nOriginal question: {full_question}"),
            ]

            # Query this speaker sequentially (debate is sequential)
            from quorate.api import query_with_fallback
            from quorate.config import api_keys
            import httpx

            keys = api_keys()
            async with httpx.AsyncClient() as client:
                name, model_used, response = await query_with_fallback(
                    client, keys, entry, msgs, max_tokens=2048, timeout=timeout, effort=effort
                )

            if not is_error(response):
                conversation.append((name, response))
                role = " [yellow](challenger)[/yellow]" if idx == challenger_idx else ""
                console.print(Panel(
                    Markdown(response), title=f"[bold]{name}[/bold]{role}", border_style="dim"
                ))
                previous_speakers = (
                    f"{previous_speakers}, {name}" if previous_speakers else name
                )
            else:
                console.print(f"[red]{name}: {response}[/red]")

    # --- JUDGE ---
    console.print("\n[bold cyan]JUDGE SYNTHESIS[/bold cyan]")
    all_text = blind_summary
    if conversation:
        debate_text = "\n\n".join(
            f"**{name}**: {text}" for name, text in conversation
        )
        all_text += f"\n\n--- DEBATE ---\n\n{debate_text}"

    judge_messages = [
        Message.system(JUDGE_SYSTEM),
        Message.user(f"Question: {full_question}\n\n{all_text}"),
    ]
    judge_response = await query_judge(
        judge, judge_messages, max_tokens=16384, timeout=300, effort=ReasoningEffort.HIGH
    )

    if is_error(judge_response):
        console.print(f"[red]Judge failed: {judge_response}[/red]")
        return ""

    console.print(Panel(Markdown(judge_response), title="[bold green]Judge[/bold green]", border_style="green"))

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

    duration = time.monotonic() - start
    console.print(f"\n[dim]({duration:.1f}s)[/dim]")

    return judge_response
