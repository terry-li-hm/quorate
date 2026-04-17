"""CLI entry point using cyclopts — subcommand per mode."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
from porin import CommandTree, EXIT_ERROR, action, emit_err, emit_ok
from rich.console import Console

from quorate import __version__


def _is_agent() -> bool:
    """True when stdout is a pipe/redirect (agent consuming), not a TTY."""
    return not sys.stdout.isatty()


# --- Command tree (bare invocation → agent self-discovery) ---

_tree = CommandTree("quorate")
_tree.add_command("council", description="Full deliberation — blind phase, debate, judge synthesis, critique", params=[
    {"name": "question", "type": "string", "required": True},
    {"name": "--context", "type": "string", "description": "Context file(s), repeatable"},
    {"name": "--deep", "type": "boolean", "default": False, "description": "Force 2+ debate rounds"},
    {"name": "--json", "type": "boolean", "default": False, "description": "Force JSON in TTY (auto in pipes)"},
], annotations={"readonly": True})
_tree.add_command("quick", description="Parallel queries — all models answer independently", params=[
    {"name": "question", "type": "string", "required": True},
    {"name": "--context", "type": "string", "description": "Context file(s), repeatable"},
    {"name": "--json", "type": "boolean", "default": False},
], annotations={"readonly": True})
_tree.add_command("redteam", description="Adversarial stress-test — find what breaks", params=[
    {"name": "question", "type": "string", "required": True},
    {"name": "--context", "type": "string", "description": "Context file(s), repeatable"},
], annotations={"readonly": True})
_tree.add_command("premortem", description="Assume failure, write past-tense narratives", params=[
    {"name": "question", "type": "string", "required": True},
    {"name": "--context", "type": "string", "description": "Context file(s), repeatable"},
], annotations={"readonly": True})
_tree.add_command("oxford", description="Binary debate — structured FOR vs AGAINST", params=[
    {"name": "question", "type": "string", "required": True},
    {"name": "--context", "type": "string", "description": "Context file(s), repeatable"},
], annotations={"readonly": True})
_tree.add_command("discuss", description="Open roundtable — no judge, conversational", params=[
    {"name": "question", "type": "string", "required": True},
    {"name": "--context", "type": "string", "description": "Context file(s), repeatable"},
], annotations={"readonly": True})


app = cyclopts.App(
    name="quorate",
    help="Multi-model deliberation CLI — frontier LLMs debate, then judge.",
    version=__version__,
)


# --- Shared parameter resolution ---


def _resolve_question(question: str | None) -> str:
    if question:
        path = Path(question)
        if path.is_file():
            return path.read_text().strip()
        return question
    # No question — emit command tree for agent discovery
    if _is_agent():
        emit_ok("quorate", _tree.to_dict())
        raise SystemExit(0)
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


def _emit_result(command: str, result: str | dict | None, json_output: bool) -> None:
    """Emit result as porin envelope when in JSON mode."""
    if not json_output:
        return
    if not result:
        emit_err(command, "No result produced", EXIT_ERROR,
                 fix="Check model availability with: quorate quick 'test'")
        return
    data = result if isinstance(result, dict) else {"response": result}
    emit_ok(command, data, [
        action("quorate redteam --context <file>", "Stress-test the synthesis"),
    ])


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
    context: tuple[str, ...] = (),
    deep: bool = False,
    json_output: Annotated[bool, cyclopts.Parameter(name="--json")] = False,
) -> None:
    """Auto-classify and deliberate (default when no subcommand given)."""
    json_output = json_output or _is_agent()
    text = _resolve_question(question)
    resolved_ctx = _resolve_context(context)
    mode = asyncio.run(_classify(text))
    if not json_output:
        Console().print(f"[dim]→ {mode}[/dim]\n")
    ctx_tuple = (resolved_ctx,) if resolved_ctx else ()
    handler = {"quick": quick, "council": council, "redteam": _preset_cmd("redteam")}.get(mode, council)
    handler(question=text, context=ctx_tuple, deep=deep, json_output=json_output)


@app.command
def quick(
    question: str | None = None,
    *,
    context: tuple[str, ...] = (),
    json_output: Annotated[bool, cyclopts.Parameter(name="--json")] = False,
) -> None:
    """Parallel queries — all models answer independently."""
    json_output = json_output or _is_agent()
    from quorate.modes.quick import run_quick
    text = _resolve_question(question)
    resolved_ctx = _resolve_context(context)
    console = Console(quiet=json_output)
    result = asyncio.run(run_quick(
        text, context=resolved_ctx, console=console,
        json_output=json_output,
    ))
    _emit_result("quorate quick", result, json_output)


@app.command
def council(
    question: str | None = None,
    *,
    context: tuple[str, ...] = (),
    deep: bool = False,
    json_output: Annotated[bool, cyclopts.Parameter(name="--json")] = False,
) -> None:
    """Full deliberation — blind phase, debate, judge synthesis, critique."""
    json_output = json_output or _is_agent()
    from quorate.modes.council import run_council
    text = _resolve_question(question)
    resolved_ctx = _resolve_context(context)
    rounds = 2 if deep else 1
    console = Console(quiet=json_output)
    result = asyncio.run(run_council(
        text, context=resolved_ctx, rounds=rounds,
        console=console, json_output=json_output,
    ))
    _emit_result("quorate council", result, json_output)


def _preset_cmd(name: str):
    """Create a handler function for a preset."""
    def handler(
        question: str | None = None,
        context: str | None = None,
        json_output: bool = False,
        **_kwargs,
    ) -> None:
        preset = PRESETS[name]
        prefix = preset["context_prefix"]
        full_context = f"{prefix}\n\n{context}" if context else prefix
        council(
            question=question, context=(full_context,),
            json_output=json_output,
        )
    return handler


# Register preset subcommands
for _name, _cfg in PRESETS.items():
    _fn = _preset_cmd(_name)
    _fn.__name__ = _name
    _fn.__doc__ = _cfg["description"]

    @app.command(name=_name)
    def _cmd(
        question: str | None = None,
        *,
        context: tuple[str, ...] = (),
        json_output: Annotated[bool, cyclopts.Parameter(name="--json")] = False,
        _preset_name: str = _name,
    ) -> None:
        resolved_ctx = _resolve_context(context)
        _preset_cmd(_preset_name)(
            question=question, context=resolved_ctx,
            json_output=json_output,
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
