"""Terminal outcomes shared by the deliberation modes and both CLI presentations."""

from collections.abc import Awaitable, Callable
from typing import Literal, TypedDict

from porin import EXIT_ERROR


def guard_judge_route[**P](runner: Callable[P, Awaitable[str]]) -> Callable[P, Awaitable[str]]:
    """Turn a route exception into a failed attempt for the phase's fallback policy."""

    async def attempt(*args: P.args, **kwargs: P.kwargs) -> str:
        try:
            return await runner(*args, **kwargs)
        except Exception:
            # Never expose provider exception text; cancellation is not a route failure.
            return "[Error: judge route failed]"

    return attempt


class Terminal(TypedDict):
    status: Literal["complete", "partial", "failed"]
    failed_phase: str | None
    reason_code: str
    exit_code: int


def finished(*, missing_contributions: bool) -> Terminal:
    return {
        "status": "partial" if missing_contributions else "complete",
        "failed_phase": None,
        "reason_code": "missing_contributions" if missing_contributions else "complete",
        "exit_code": 0,
    }


def failed(phase: str, reason: str) -> Terminal:
    return {
        "status": "failed",
        "failed_phase": phase,
        "reason_code": reason,
        "exit_code": EXIT_ERROR,
    }
