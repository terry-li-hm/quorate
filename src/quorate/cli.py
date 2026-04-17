"""CLI entry point using cyclopts — subcommand per mode."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
from porin import EXIT_ERROR, action, emit_err, emit_ok
from rich.console import Console


def _is_agent() -> bool:
    """True when stdout is a pipe/redirect (agent consuming), not a TTY."""
    return not sys.stdout.isatty()

from quorate import __version__

app = cyclopts.App(
    name="quorate",
    help="Multi-model deliberation CLI — frontier LLMs debate, then judge.",
    version=__version__,
)


# --- Shared parameter resolution ---


def _resolve_question(question: str | None, prompt_file: Path | None) -> str:
    if prompt_file:
        return prompt_file.read_text().strip()
    if question:
        # If it looks like a file path that exists, read it
        path = Path(question)
        if path.is_file():
            return path.read_text().strip()
        return question
    Console().print("[red]No question provided.[/red]")
    raise SystemExit(1)


def _resolve_context(context: tuple[str, ...]) -> str | None:
    """Resolve context: each item can be a file path or inline text. Multiple items concatenated."""
    if not context:
        return None
    parts = []
    for item in context:
        path = Path(item).expanduser()
        if path.is_file():
            parts.append(path.read_text().strip())
        else:
            parts.append(item)
    return "\n\n---\n\n".join(parts)


def _write_output(
    command: str, output: Path | None, result: str | dict | None,
    console: Console, json_output: bool,
) -> None:
    """Write result to file or stdout using porin envelope."""
    if not result:
        if json_output:
            emit_err(command, "No result produced", EXIT_ERROR,
                     fix="Check model availability with: quorate quick 'test'")
        return
    if json_output:
        data = result if isinstance(result, dict) else {"response": result}
        next_actions = [
            action(f"quorate redteam --context <file>", "Stress-test the synthesis"),
        ]
        if not output:
            emit_ok(command, data, next_actions)
        else:
            from porin import ok
            envelope = ok(command, data, next_actions)
            output.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n")
            Console().print(f"[dim]→ {output}[/dim]")
    elif output:
        output.write_text(console.export_text())
        Console().print(f"[dim]→ {output}[/dim]")


def _resolve_effort(effort: str | None):
    if not effort:
        return None
    from quorate.config import ReasoningEffort
    try:
        return ReasoningEffort(effort.lower())
    except ValueError:
        Console().print(f"[red]Invalid effort: {effort}. Use low/medium/high.[/red]")
        raise SystemExit(1)


# --- Presets: named council configurations ---

PRESETS = {
    "redteam": {
        "description": "Adversarial stress-test — find what breaks.",
        "context_prefix": (
            "RED TEAM EXERCISE: Your job is to BREAK this plan, not improve it. "
            "Every attack must be specific: 'When X happens, Y fails because Z.' "
            "Find specific, concrete failure modes. Be adversarial, not constructive."
        ),
        "rounds": 1,
    },
    "premortem": {
        "description": "Assume failure, write past-tense narratives.",
        "context_prefix": (
            "PRE-MORTEM EXERCISE: Assume this plan has ALREADY FAILED. It is 12 months later. "
            "Write in first person, past tense: 'Here's what happened.' "
            "No hedging words (might, could, possibly). Give a specific chain of events: "
            "trigger → escalation → breakdown → consequence. "
            "Include one decision we got wrong and one signal we ignored."
        ),
        "rounds": 1,
    },
    "oxford": {
        "description": "Binary debate — structured FOR vs AGAINST.",
        "context_prefix": (
            "OXFORD DEBATE: This is a binary motion. Two sides will argue FOR and AGAINST. "
            "You are assigned a side regardless of your personal view. Argue it convincingly. "
            "Present 2-3 clear arguments with evidence. Concede what you must — "
            "selective concession is persuasive, blanket denial is not."
        ),
        "rounds": 1,
    },
    "discuss": {
        "description": "Open roundtable — no judge, conversational.",
        "context_prefix": (
            "ROUNDTABLE DISCUSSION: This is an open exploration, not a debate. "
            "Riff on what others said — 'that reminds me of...' or 'the interesting thing is...' "
            "Share analogies and surprising angles. No bullet points. "
            "Think conference after-party, not panel presentation."
        ),
        "rounds": 1,
        "no_judge": True,
        "no_critic": True,
    },
}


# --- Subcommands ---


@app.default
def auto(
    question: str | None = None,
    *,
    prompt_file: Path | None = None,
    context: tuple[str, ...] = (),
    timeout: float = 300,
    effort: str | None = None,
    quiet: bool = False,
    json_output: Annotated[bool, cyclopts.Parameter(name="--json")] = False,
    output: Path | None = None,
) -> None:
    """Auto-classify and deliberate (default when no subcommand given)."""
    json_output = json_output or _is_agent()
    text = _resolve_question(question, prompt_file)
    resolved_ctx = _resolve_context(context)
    mode = asyncio.run(_classify(text))
    if not json_output:
        Console(quiet=quiet).print(f"[dim]→ {mode}[/dim]\n")
    # Re-dispatch — wrap resolved context back into tuple for CLI functions
    ctx_tuple = (resolved_ctx,) if resolved_ctx else ()
    handler = {"quick": quick, "council": council, "redteam": _preset_cmd("redteam")}.get(mode, council)
    handler(question=text, context=ctx_tuple, timeout=timeout, effort=effort, quiet=quiet, json_output=json_output, output=output)


@app.command
def quick(
    question: str | None = None,
    *,
    prompt_file: Path | None = None,
    context: tuple[str, ...] = (),
    timeout: float = 300,
    effort: str | None = None,
    quiet: bool = False,
    json_output: Annotated[bool, cyclopts.Parameter(name="--json")] = False,
    output: Path | None = None,
) -> None:
    """Parallel queries — all models answer independently."""
    json_output = json_output or _is_agent()
    from quorate.modes.quick import run_quick
    text = _resolve_question(question, prompt_file)
    resolved_ctx = _resolve_context(context)
    console = Console(record=bool(output), quiet=quiet or json_output)
    result = asyncio.run(run_quick(
        text, context=resolved_ctx, timeout=timeout,
        effort=_resolve_effort(effort), console=console,
        json_output=json_output,
    ))
    _write_output("quorate quick", output, result, console, json_output)


@app.command
def council(
    question: str | None = None,
    *,
    prompt_file: Path | None = None,
    context: tuple[str, ...] = (),
    rounds: int = 1,
    deep: bool = False,
    timeout: float = 300,
    effort: str | None = None,
    judge_model: str | None = None,
    critic_model: str | None = None,
    no_critic: bool = False,
    no_judge: bool = False,
    domain: str | None = None,
    persona: str | None = None,
    quiet: bool = False,
    json_output: Annotated[bool, cyclopts.Parameter(name="--json")] = False,
    output: Path | None = None,
) -> None:
    """Full deliberation — blind phase, debate, judge synthesis, critique."""
    json_output = json_output or _is_agent()
    from quorate.modes.council import run_council
    text = _resolve_question(question, prompt_file)
    resolved_ctx = _resolve_context(context)
    if deep:
        rounds = max(rounds, 2)
    console = Console(record=bool(output), quiet=quiet or json_output)
    result = asyncio.run(run_council(
        text, context=resolved_ctx, rounds=rounds, timeout=timeout,
        effort=_resolve_effort(effort), judge_model=judge_model,
        critic_model=critic_model, no_critic=no_critic or no_judge,
        domain=domain, persona=persona, console=console,
        json_output=json_output,
    ))
    _write_output("quorate council", output, result, console, json_output)


def _preset_cmd(name: str):
    """Create a handler function for a preset."""
    def handler(
        question: str | None = None,
        prompt_file: Path | None = None,
        context: str | None = None,
        rounds: int | None = None,
        timeout: float = 300,
        effort: str | None = None,
        quiet: bool = False,
        json_output: bool = False,
        output: Path | None = None,
        **_kwargs,
    ) -> None:
        preset = PRESETS[name]
        prefix = preset["context_prefix"]
        full_context = f"{prefix}\n\n{context}" if context else prefix
        council(
            question=question, prompt_file=prompt_file, context=(full_context,),
            rounds=rounds or preset.get("rounds", 1), timeout=timeout, effort=effort,
            no_critic=preset.get("no_critic", False),
            no_judge=preset.get("no_judge", False), quiet=quiet,
            json_output=json_output, output=output,
        )
    return handler


# Register preset subcommands
for _name, _cfg in PRESETS.items():
    _fn = _preset_cmd(_name)
    _fn.__name__ = _name
    _fn.__doc__ = _cfg["description"]
    # Cyclopts command registration with proper signature
    @app.command(name=_name)
    def _cmd(
        question: str | None = None,
        *,
        prompt_file: Path | None = None,
        context: tuple[str, ...] = (),
        rounds: int | None = None,
        timeout: float = 300,
        effort: str | None = None,
        quiet: bool = False,
        json_output: Annotated[bool, cyclopts.Parameter(name="--json")] = False,
        output: Path | None = None,
        _preset_name: str = _name,
    ) -> None:
        resolved_ctx = _resolve_context(context)
        _preset_cmd(_preset_name)(
            question=question, prompt_file=prompt_file, context=resolved_ctx,
            rounds=rounds, timeout=timeout, effort=effort, quiet=quiet,
            json_output=json_output, output=output,
        )
    _cmd.__doc__ = _cfg["description"]


# --- Auto-classify ---


async def _classify(question: str) -> str:
    from quorate.api import query_judge
    from quorate.config import CLASSIFIER_MODEL, Message
    from quorate.prompts import CLASSIFIER_PROMPT

    messages = [Message.system(CLASSIFIER_PROMPT), Message.user(question)]
    response = await query_judge(CLASSIFIER_MODEL, messages, max_tokens=10, timeout=15)
    result = response.strip().lower().rstrip(".")
    valid = {"quick", "council", "redteam"}
    return result if result in valid else "council"
