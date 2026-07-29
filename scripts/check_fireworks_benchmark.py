#!/usr/bin/env python3
"""Validate the Fireworks US seat in a Quorate benchmark envelope."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

FIREWORKS_US_MODEL = "accounts/fireworks/routers/kimi-k3-us"
FIREWORKS_US_CLIENT_CONTRACT = "fireworks-us-k3-priority-v1"
FIREWORKS_US_SERVICE_TIER = "priority"


def evaluate(payload: dict[str, Any]) -> str:
    """Return absent, pass, or fail for the exact Fireworks US route."""
    report = payload["result"]
    row = next(
        (model for model in report["models"] if model.get("model_id") == FIREWORKS_US_MODEL),
        None,
    )
    if row is None:
        return "absent"
    valid = (
        row.get("canaries") == 3
        and row.get("reachable") == 3
        and row.get("passed") == 3
        and row.get("providers") == {"fireworks-us": 3}
        and not row.get("diagnostics")
        and row.get("request_contract")
        == {
            "client_contract": FIREWORKS_US_CLIENT_CONTRACT,
            "requested_service_tier": FIREWORKS_US_SERVICE_TIER,
        }
    )
    return "pass" if valid else "fail"


def main() -> int:
    arguments = sys.argv[1:]
    required = False
    if arguments[:1] == ["--required"]:
        required = True
        arguments = arguments[1:]
    if len(arguments) != 1:
        return 3
    try:
        payload = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
        status = evaluate(payload)
    except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError):
        return 3
    if required and status == "absent":
        return 1
    return {"pass": 0, "fail": 1, "absent": 2}[status]


if __name__ == "__main__":
    raise SystemExit(main())
