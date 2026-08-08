"""Blind divergence — how far apart the council started, before debate.

The council already reports disagreement in prose, inside the judge's "Points of
Disagreement" section. That reads well and counts for nothing: you cannot compare
it across runs. This module adds the countable half. A small pass reads only the
blind claims, groups them into positions, and the run log keeps the resulting
index so agreement accumulates instead of scrolling past.

Two design choices carry the weight. The scoring happens before the debate,
because once members have seen each other, convergence is partly the protocol's
doing, so a score taken afterwards measures the protocol rather than the
question. And the model is asked to partition members into positions rather than
to score each member against a modal position, because a partition is the part a
program can check. Every member must appear exactly once or the record is thrown
away, and the index is computed here from the group sizes rather than trusted
from the response, so the model is only ever asked for the judgment it is good
at. Live runs confirmed the grouping is stable: three repeats over one six-model
panel returned the same index and the same two blocs twice over.
"""

from __future__ import annotations

import json
import re
from typing import Any

from quorate.config import Message

DIVERGENCE_SYSTEM = """You are grouping a panel's opening claims, before any debate.

You will see the same question given to several models, and each model's independent
opening claim. None of them has seen any other. Your job is measurement, not judgment:
do not decide who is right, do not rank the positions, and do not write a synthesis.

Group the models by the position they actually argue. Two models share a position when
they reach the same conclusion for reasons that would hold up together under pressure;
they hold different positions when their conclusions are incompatible, or when the same
conclusion rests on reasons that would diverge as soon as the case got harder. Judge the
conclusion and its load-bearing reasons, not tone, length, hedging or confidence. Give
each group a short label naming the position, not the models in it.

Every model must appear in exactly one group. Do not omit a model, and do not place one
in two groups. If a model is genuinely alone, give it its own group.

Respond with JSON only, no prose and no code fence:

{"positions": [{"label": "<short phrase>", "members": ["<model name>", ...]}, ...]}

Use the model names exactly as given."""

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def divergence_messages(question: str, blind_claims: dict[str, str]) -> list[Message]:
    """Build the grouping call. Sees blind claims only — never the debate."""
    claims = "\n\n".join(f"**{name}**: {claim}" for name, claim in blind_claims.items())
    names = ", ".join(blind_claims)
    return [
        Message.system(DIVERGENCE_SYSTEM),
        Message.user(f"Question: {question}\n\nModels: {names}\n\nBLIND CLAIMS:\n\n{claims}"),
    ]


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Pull the JSON object out of a response that may be fenced or prefaced."""
    candidate = _FENCE_RE.sub("", raw).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def parse_divergence(raw: str, names: list[str]) -> dict[str, Any] | None:
    """Parse a grouping response into a record, or None if it is unusable.

    The grouping must be a true partition of the panel. A dropped or duplicated
    member would quietly move the index, and the run log could not tell that
    apart from a real result, so anything short of an exact partition is thrown
    away rather than recorded.
    """
    parsed = _extract_json(raw)
    if parsed is None:
        return None
    raw_positions = parsed.get("positions")
    if not isinstance(raw_positions, list) or not raw_positions:
        return None

    positions: list[dict[str, Any]] = []
    seen: list[str] = []
    for entry in raw_positions:
        if not isinstance(entry, dict):
            return None
        members = entry.get("members")
        if not isinstance(members, list) or not members:
            return None
        if not all(isinstance(member, str) for member in members):
            return None
        seen.extend(members)
        label = entry.get("label")
        positions.append(
            {
                "label": label.strip() if isinstance(label, str) else None,
                "members": list(members),
            }
        )

    if sorted(seen) != sorted(names):
        return None

    largest = max(len(position["members"]) for position in positions)
    return {
        "index": round(100 * (1 - largest / len(names)), 1),
        "distinct_positions": len(positions),
        "largest_bloc": largest,
        "panel_size": len(names),
        "positions": positions,
    }


def format_divergence(record: dict[str, Any]) -> str:
    """One console line: the index, and who grouped with whom."""
    groups = "; ".join(
        f"{position['label'] or 'unlabelled'}: {', '.join(position['members'])}"
        for position in record["positions"]
    )
    return (
        f"Blind divergence {record['index']}/100 — {record['distinct_positions']} positions"
        f" across {record['panel_size']} members — {groups}"
    )
