"""Exercise both presentations through the command handlers, without live routes."""

import asyncio
import io
import json
import sys

import pytest
from porin import EXIT_ERROR

from quorate import cli, local
from quorate.config import ModelCallResult, ModelEntry
from quorate.divergence import DIVERGENCE_SYSTEM
from quorate.prompts import CRITIQUE_SYSTEM

ROSTER = [ModelEntry(f"Seat-{i}", f"synthetic/{i}") for i in range(3)]
INITIAL = {"quick": "responses", "brainstorm": "generation", "council": "blind"}
FINAL = {"council": "judge", "brainstorm": "curation"}


class LocalStub:
    def __init__(self, successes, scenario, raises=False):
        self.successes = successes
        self.scenario = scenario
        self.raises = raises
        self.calls = []

    def route_failure(self, returned="[Error: route unavailable]"):
        if self.raises:
            raise ValueError("PRIVATE_ROUTE_DETAIL")
        return returned

    async def run_parallel(self, entries, *args, **kwargs):
        self.calls.append("initial")
        return [
            ModelCallResult(
                e.name, e.model, "Seed evidence" if i < self.successes else "[Error: seat]"
            )
            for i, e in enumerate(entries)
        ]

    async def run_prompt_round(self, entries, *args, **kwargs):
        if not self.calls:
            return await self.run_parallel(entries)
        return [await self.query_model(e) for e in entries]

    async def query_model(self, entry, *args, **kwargs):
        self.calls.append("discussion")
        response = (
            self.route_failure()
            if self.scenario == "lost_turn" and entry == ROSTER[0]
            else "More evidence"
        )
        return ModelCallResult(entry.name, entry.model, response)

    async def query_judge(self, model, messages, *, max_tokens, **kwargs):
        if messages[0].content in (DIVERGENCE_SYSTEM, CRITIQUE_SYSTEM):
            self.calls.append("optional")
            return self.route_failure()
        self.calls.append(model)
        if self.scenario == "exhaust" or (
            self.scenario == "fallback" and model == local.CODEX_ENTRY.model
        ):
            return self.route_failure("[Error: PRIVATE_ROUTE_DETAIL]")
        return "## Shortlist\nProduct\n## Wildcard\nIdea\n## Pattern map\nThemes"


@pytest.fixture
def offline_cli(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr(local, "local_models", lambda: ROSTER)


@pytest.fixture
def presentations(offline_cli, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "_is_agent", lambda: False)
    results = []
    emit = cli._emit_result

    def observe(command, result, json_output):
        results.append(result)
        return emit(command, result, json_output)

    monkeypatch.setattr(cli, "_emit_result", observe)

    def run(mode, successes, scenario="primary", *, raises=False):
        for machine in (False, True):
            stub = LocalStub(successes, scenario, raises=raises)
            monkeypatch.setattr(cli, "_resolve_local_runner", lambda *_: stub)
            code = 0
            try:
                getattr(cli, mode)(
                    "Synthetic question", local=True, workspace=".", json_output=machine
                )
            except SystemExit as exc:
                code = exc.code
            result = results[-1]
            terminal = result["terminal"]
            assert code == terminal["exit_code"]
            streams = capsys.readouterr()
            assert "PRIVATE_ROUTE_DETAIL" not in streams.out + streams.err
            assert "PRIVATE_ROUTE_DETAIL" not in json.dumps(result)
            assert b"PRIVATE_ROUTE_DETAIL" not in (tmp_path / "quorate" / "runs.jsonl").read_bytes()
            if machine:
                *events, envelope = map(json.loads, streams.out.splitlines())
                assert all("ok" not in event for event in events)
                assert envelope["ok"] is (code == 0)
                assert envelope["result"] == result
                assert not streams.err
            else:
                assert '"ok":' not in streams.out + streams.err
                if code:
                    assert terminal["failed_phase"] in streams.err
                    assert f"{successes} successful" in streams.err
                elif terminal["status"] == "partial":
                    assert "Partial:" in streams.out
            if successes < 2 or mode == "quick":
                assert stub.calls == ["initial"]
        assert results[-2]["terminal"] == results[-1]["terminal"]
        return result, stub.calls

    return run


@pytest.mark.parametrize("mode", INITIAL)
@pytest.mark.parametrize(
    "successes,status", [(3, "complete"), (2, "partial"), (1, "failed"), (0, "failed")]
)
def test_q3_terminal_initial_quorum(presentations, mode, successes, status):
    result, _ = presentations(mode, successes)
    phase = INITIAL[mode]
    assert result["quorum_achieved"] is (successes >= 2)
    assert (result["success_count"], result["failed_count"], result["quorum_target"]) == (
        successes,
        3 - successes,
        2,
    )
    rows = result["responses"] if mode == "quick" else result["phases"][phase]
    assert [r["model_id"] for r in rows] == [e.model for e in ROSTER]
    assert [r.get("response", r.get("error")) for r in rows] == ["Seed evidence"] * successes + [
        "[Error: seat]"
    ] * (3 - successes)
    terminal = result["terminal"]
    assert terminal["status"] == status and terminal["reason_code"]
    assert terminal["failed_phase"] == (phase if successes < 2 else None)
    assert terminal["exit_code"] == (EXIT_ERROR if successes < 2 else 0)


@pytest.mark.parametrize("mode", FINAL)
@pytest.mark.parametrize("scenario", ["exhaust", "fallback"])
@pytest.mark.parametrize("raises", [False, True], ids=["returned", "raised"])
def test_q3_terminal_final_product(presentations, mode, scenario, raises):
    result, calls = presentations(mode, 3, scenario, raises=raises)
    phase = FINAL[mode]
    assert result["quorum_achieved"] and result["success_count"] == 3
    assert [r["response"] for r in result["phases"][INITIAL[mode]]] == ["Seed evidence"] * 3
    turns = result["phases"]["debate" if mode == "council" else "cross_pollination"]
    assert [r["model_id"] for r in turns] == [e.model for e in ROSTER]
    assert all(r["response"] == "More evidence" for r in turns)
    assert calls.count("discussion") == 3
    product = result["phases"][phase]
    terminal = result["terminal"]
    if scenario == "exhaust":
        assert terminal["status"] == "failed" and terminal["exit_code"] == EXIT_ERROR
        assert terminal["failed_phase"] == phase
        assert calls[-2:] == [local.CODEX_ENTRY.model, local.GEMINI_ENTRY.model]
        assert "error" in product and "response" not in product
    else:
        assert terminal["exit_code"] == 0 and product["response"]
        assert terminal["status"] == "complete"
        assert product["model"] == local.GEMINI_ENTRY.model
        assert product["preferred_model"] == local.CODEX_ENTRY.model
        if mode == "council":
            assert "critique" not in result["phases"] and "divergence" not in result
            assert calls.count("optional") == 4


@pytest.mark.parametrize(
    "mode,raises", [("council", True), ("council", False), ("brainstorm", False)]
)
def test_q3_terminal_lost_turn(presentations, mode, raises):
    result, _ = presentations(mode, 3, "lost_turn", raises=raises)
    assert result["terminal"]["status"] == "partial" and result["terminal"]["exit_code"] == 0
    assert result["phases"][FINAL[mode]]["response"]
    first, *rest = result["phases"]["debate" if mode == "council" else "cross_pollination"]
    assert first["model_id"] == ROSTER[0].model and "error" in first
    if raises:
        assert first["diagnostics"] == ["orchestrator:exception"]
    assert [r["response"] for r in rest] == ["More evidence"] * 2


def test_q3_terminal_json_on_tty_never_prompts(offline_cli, monkeypatch, capsys):
    stub = LocalStub(3, "primary")
    monkeypatch.setattr(cli, "_resolve_local_runner", lambda *_: stub)
    stdin = io.StringIO("m\ns\n")
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    cli.council("Synthetic question", local=True, workspace=".", json_output=True)
    assert stdin.tell() == 0
    streams = capsys.readouterr()
    assert not streams.err
    *events, envelope = map(json.loads, streams.out.splitlines())
    assert all("type" in event and "ok" not in event for event in events)
    assert envelope["ok"] and envelope["result"]["terminal"]["status"] == "complete"


@pytest.mark.parametrize(
    "mode,route",
    [("council", "query_judge"), ("brainstorm", "query_judge"), ("council", "query_model")],
)
def test_q3_terminal_cancellation_propagates(offline_cli, monkeypatch, mode, route):
    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    stub = LocalStub(3, "primary")
    monkeypatch.setattr(stub, route, cancel)
    monkeypatch.setattr(cli, "_resolve_local_runner", lambda *_: stub)
    options = {"divergence": False} if mode == "council" else {}
    with pytest.raises(asyncio.CancelledError):
        getattr(cli, mode)(
            "Synthetic question", local=True, workspace=".", json_output=True, **options
        )
