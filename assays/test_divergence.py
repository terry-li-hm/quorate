"""Tests for blind divergence grouping and its parser."""

from quorate.divergence import divergence_messages, format_divergence, parse_divergence

NAMES = ["Alpha", "Beta", "Gamma", "Delta"]


def _grouping(*groups: list[str]) -> str:
    positions = [
        {"label": f"Position {index}", "members": members} for index, members in enumerate(groups)
    ]
    return __import__("json").dumps({"positions": positions})


def test_a_unanimous_panel_scores_zero():
    record = parse_divergence(_grouping(NAMES), NAMES)
    assert record is not None
    assert record["index"] == 0.0
    assert record["distinct_positions"] == 1


def test_an_evenly_split_panel_scores_by_the_largest_bloc():
    record = parse_divergence(_grouping(NAMES[:2], NAMES[2:]), NAMES)
    assert record is not None
    assert record["index"] == 50.0
    assert record["largest_bloc"] == 2


def test_an_uneven_split_scores_below_an_even_one():
    # A lopsided panel is closer to agreement than a split one, and the index
    # has to say so: four-two reads lower than the three-three case above.
    names = [*NAMES, "Epsilon", "Zeta"]
    record = parse_divergence(_grouping(names[:4], names[4:]), names)
    assert record is not None
    assert record["index"] == 33.3
    assert record["largest_bloc"] == 4


def test_total_disagreement_approaches_the_ceiling():
    record = parse_divergence(_grouping(*([name] for name in NAMES)), NAMES)
    assert record is not None
    assert record["index"] == 75.0
    assert record["distinct_positions"] == 4


def test_parse_accepts_a_fenced_response():
    record = parse_divergence(f"```json\n{_grouping(NAMES[:1], NAMES[1:])}\n```", NAMES)
    assert record is not None
    assert record["index"] == 25.0


def test_parse_accepts_json_after_a_preamble():
    record = parse_divergence(f"Here is the grouping:\n{_grouping(NAMES)}", NAMES)
    assert record is not None
    assert record["positions"][0]["label"] == "Position 0"


def test_parse_rejects_a_dropped_member():
    # Dropping a member would quietly shrink the panel and move the index.
    assert parse_divergence(_grouping(NAMES[:2], NAMES[2:3]), NAMES) is None


def test_parse_rejects_a_duplicated_member():
    assert parse_divergence(_grouping(NAMES[:2], [NAMES[1], *NAMES[2:]]), NAMES) is None


def test_parse_rejects_an_invented_member():
    assert parse_divergence(_grouping(NAMES, ["Omega"]), NAMES) is None


def test_parse_rejects_malformed_and_empty_responses():
    assert parse_divergence("I cannot group these.", NAMES) is None
    assert parse_divergence('{"positions": []}', NAMES) is None
    assert parse_divergence('{"positions": [{"members": []}]}', NAMES) is None
    assert parse_divergence('{"positions": [{"members": "Alpha"}]}', NAMES) is None
    assert parse_divergence('["Alpha"]', NAMES) is None


def test_grouping_prompt_carries_only_blind_claims():
    messages = divergence_messages("Should we ship?", {"Alpha": "Yes", "Beta": "No"})
    assert "BLIND CLAIMS" in messages[1].content
    assert "Alpha" in messages[1].content and "Beta" in messages[1].content
    assert "measurement, not judgment" in messages[0].content


def test_format_reports_the_index_and_the_groups():
    record = parse_divergence(_grouping(NAMES[:3], NAMES[3:]), NAMES)
    assert record is not None
    line = format_divergence(record)
    assert "Blind divergence 25.0/100" in line
    assert "2 positions across 4 members" in line
    assert "Position 0: Alpha, Beta, Gamma" in line
