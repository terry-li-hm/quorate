"""CLI entry point using cyclopts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import cyclopts
from rich.console import Console

from quorate import __version__

app = cyclopts.App(
    name="quorate",
    help="Multi-model deliberation CLI — frontier LLMs debate, then judge.",
    version=__version__,
)


@app.default
def main(
    question: str | None = None,
    *,
    # Mode flags
    quick: bool = False,
    council: bool = False,
    redteam: bool = False,
    oxford: bool = False,
    deep: bool = False,
    # Context
    context: str | None = None,
    persona: str | None = None,
    domain: str | None = None,
    prompt_file: Path | None = None,
    # Deliberation
    rounds: int = 1,
    effort: str | None = None,
    # Models
    judge_model: str | None = None,
    critic_model: str | None = None,
    no_critic: bool = False,
    # Output
    timeout: float = 120,
    quiet: bool = False,
) -> None:
    """Run a multi-model deliberation."""
    console = Console(quiet=quiet)

    # Resolve question
    if prompt_file:
        question = prompt_file.read_text().strip()
    if not question:
        console.print("[red]No question provided.[/red]")
        raise SystemExit(1)

    # Resolve effort
    effort_enum = None
    if effort:
        from quorate.config import ReasoningEffort
        try:
            effort_enum = ReasoningEffort(effort.lower())
        except ValueError:
            console.print(f"[red]Invalid effort: {effort}. Use low/medium/high.[/red]")
            raise SystemExit(1)

    # Resolve mode
    if quick:
        mode = "quick"
    elif council or deep:
        mode = "council"
        if deep:
            rounds = max(rounds, 2)
    elif redteam:
        mode = "redteam"
    elif oxford:
        mode = "oxford"
    else:
        # Auto-classify
        mode = asyncio.run(_classify(question))

    console.print(f"[dim]Mode: {mode}[/dim]\n")

    # Dispatch
    if mode == "quick":
        from quorate.modes.quick import run_quick
        asyncio.run(run_quick(
            question, context=context, timeout=timeout, effort=effort_enum, console=console
        ))

    elif mode == "council":
        from quorate.modes.council import run_council
        asyncio.run(run_council(
            question,
            context=context,
            rounds=rounds,
            timeout=timeout,
            effort=effort_enum,
            judge_model=judge_model,
            critic_model=critic_model,
            no_critic=no_critic,
            domain=domain,
            persona=persona,
            console=console,
        ))

    elif mode == "redteam":
        # Simplified: run as council with redteam framing
        console.print("[yellow]Red team mode — running as council with adversarial framing.[/yellow]\n")
        from quorate.modes.council import run_council
        redteam_context = (
            "RED TEAM EXERCISE: Your job is to BREAK this plan, not improve it. "
            "Find specific, concrete failure modes. Be adversarial."
        )
        full_context = f"{redteam_context}\n\n{context}" if context else redteam_context
        asyncio.run(run_council(
            question,
            context=full_context,
            rounds=rounds or 1,
            timeout=timeout,
            effort=effort_enum,
            console=console,
        ))

    elif mode == "oxford":
        console.print("[yellow]Oxford mode not yet ported. Running as council.[/yellow]\n")
        from quorate.modes.council import run_council
        asyncio.run(run_council(
            question, context=context, rounds=1, timeout=timeout,
            effort=effort_enum, console=console,
        ))


async def _classify(question: str) -> str:
    """Auto-classify question into best mode."""
    from quorate.api import query_judge
    from quorate.config import CLASSIFIER_MODEL, Message
    from quorate.prompts import CLASSIFIER_PROMPT

    messages = [Message.system(CLASSIFIER_PROMPT), Message.user(question)]
    response = await query_judge(CLASSIFIER_MODEL, messages, max_tokens=10, timeout=15)
    result = response.strip().lower().rstrip(".")
    valid = {"quick", "council", "oxford", "redteam", "discuss"}
    return result if result in valid else "council"
