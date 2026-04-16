"""Quick mode: parallel queries, no debate, no judge."""

from __future__ import annotations

import asyncio
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from quorate.api import run_parallel
from quorate.config import Message, ModelEntry, ReasoningEffort, is_error, quick_models


async def run_quick(
    question: str,
    context: str | None = None,
    models: list[ModelEntry] | None = None,
    timeout: float = 90,
    effort: ReasoningEffort | None = None,
    console: Console | None = None,
) -> str:
    """Run quick parallel query across all models."""
    console = console or Console()
    models = models or quick_models()
    full_question = f"{context}\n\n{question}" if context else question
    messages = [Message.user(full_question)]

    console.print(f"[dim](querying {len(models)} models in parallel...)[/dim]\n")
    start = time.monotonic()

    results = await run_parallel(models, messages, max_tokens=2048, timeout=timeout, effort=effort)

    duration = time.monotonic() - start
    transcript_parts = []

    for name, model_used, response in results:
        if is_error(response):
            console.print(f"[red]{name}: {response}[/red]")
        else:
            console.print(Panel(Markdown(response), title=f"[bold]{name}[/bold]", border_style="dim"))
            transcript_parts.append(f"### {name}\n{response}")

    console.print(f"\n[dim]({duration:.1f}s)[/dim]")
    return "\n\n".join(transcript_parts)
