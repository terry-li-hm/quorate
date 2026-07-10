"""Deterministic helpers for HeavySkill synthesis-stage trace discipline."""

from __future__ import annotations

import random
import re
import time
from typing import TypeVar

TraceT = TypeVar("TraceT")

_DROP_SENTENCE_PATTERNS = (
    re.compile(r"(?i)^\s*wait\b"),
    re.compile(r"(?i)^\s*on second thought\b"),
    re.compile(r"(?i)\blet me reconsider\b"),
    re.compile(r"(?i)\bi am wrong\b"),
    re.compile(r"(?i)\bi (?:may|might) be wrong\b"),
)
_PREFIX_PATTERNS = (
    re.compile(r"(?i)^\s*i think\b[\s,:-]*"),
    re.compile(r"(?i)^\s*maybe\b[\s,:-]*"),
    re.compile(r"(?i)^\s*perhaps\b[\s,:-]*"),
    re.compile(r"(?i)^\s*actually\b[\s,:-]*"),
)


# HeavySkill (Wang et al., ICML 2026, arXiv:2605.02396)
def shuffle_traces(traces: list[TraceT], seed: int | None = None) -> list[TraceT]:
    """Return a shuffled copy of traces, preserving the input list."""
    shuffled = list(traces)
    rng = random.Random(seed if seed is not None else time.time_ns())
    rng.shuffle(shuffled)
    return shuffled


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _should_drop_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    return any(pattern.search(stripped) for pattern in _DROP_SENTENCE_PATTERNS)


def _strip_prefixes(sentence: str) -> str:
    stripped = sentence.strip()
    changed = True
    while changed and stripped:
        changed = False
        for pattern in _PREFIX_PATTERNS:
            updated = pattern.sub("", stripped, count=1).strip()
            if updated != stripped:
                stripped = updated
                changed = True
    return stripped


# HeavySkill (Wang et al., ICML 2026, arXiv:2605.02396)
def prune_cot(text: str, max_tokens: int = 2000) -> str:
    """Drop hedge and self-questioning sentences, then cap the result by token count."""
    normalized = _normalize_whitespace(text)
    if not normalized:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    kept: list[str] = []
    for sentence in sentences:
        stripped = _strip_prefixes(sentence)
        if not stripped or _should_drop_sentence(stripped):
            continue
        kept.append(stripped)
    pruned = " ".join(kept) if kept else normalized

    words = pruned.split()
    if len(words) <= max_tokens:
        return pruned
    return " ".join(words[:max_tokens])
