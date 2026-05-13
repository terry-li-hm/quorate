"""Tests for HeavySkill trace discipline.

Reference: HeavySkill — Heavy Thinking as the Inner Skill in Agentic Harness
(Wang et al., ICML 2026, arXiv:2605.02396).

Two treatments on quorate's synthesis stage:
  1. shuffle_traces — randomise model contribution order before synthesis
     to mitigate position-anchoring bias in the judge.
  2. prune_cot — deterministic regex strip of hedge / self-questioning
     passages, preserving conclusion + key reasoning steps.

Both treatments are additive (default-off CLI flags --shuffle-traces and
--prune-cot). Implementation lives in src/quorate/heavyskill.py.
"""

from quorate.heavyskill import prune_cot, shuffle_traces


SAMPLE_TRACES = [
    {"model": "claude", "answer": "A", "reasoning": "Because X."},
    {"model": "gpt", "answer": "B", "reasoning": "Because Y."},
    {"model": "gemini", "answer": "C", "reasoning": "Because Z."},
    {"model": "glm", "answer": "A", "reasoning": "Because X'."},
    {"model": "grok", "answer": "B", "reasoning": "Because Y'."},
]


class TestShuffleTraces:
    def test_seeded_shuffle_is_deterministic(self):
        """Same seed → same order, regardless of run."""
        a = shuffle_traces(SAMPLE_TRACES, seed=42)
        b = shuffle_traces(SAMPLE_TRACES, seed=42)
        assert [t["model"] for t in a] == [t["model"] for t in b]

    def test_different_seeds_produce_different_orders(self):
        """Seeds 1 and 2 should not collide on a 5-element list (P ~= 1/120)."""
        a = shuffle_traces(SAMPLE_TRACES, seed=1)
        b = shuffle_traces(SAMPLE_TRACES, seed=2)
        assert [t["model"] for t in a] != [t["model"] for t in b]

    def test_shuffle_preserves_all_traces(self):
        """No trace is dropped or duplicated."""
        result = shuffle_traces(SAMPLE_TRACES, seed=7)
        assert len(result) == len(SAMPLE_TRACES)
        assert sorted(t["model"] for t in result) == sorted(t["model"] for t in SAMPLE_TRACES)

    def test_shuffle_does_not_mutate_input(self):
        """Original list order is preserved after call."""
        original_order = [t["model"] for t in SAMPLE_TRACES]
        shuffle_traces(SAMPLE_TRACES, seed=11)
        assert [t["model"] for t in SAMPLE_TRACES] == original_order

    def test_unseeded_shuffle_varies(self):
        """No seed → time-based, so two consecutive calls almost always differ."""
        results = set()
        for _ in range(5):
            r = shuffle_traces(SAMPLE_TRACES)
            results.add(tuple(t["model"] for t in r))
        assert len(results) > 1, "Unseeded shuffle should produce varying orders"


class TestPruneCot:
    def test_strips_hedge_phrases(self):
        """Standard hedges (I think, maybe, perhaps) are removed."""
        text = "I think the answer is 42. Maybe I am wrong. Perhaps consider this."
        pruned = prune_cot(text)
        assert "I think" not in pruned
        assert "Maybe" not in pruned
        assert "Perhaps" not in pruned
        assert "42" in pruned

    def test_strips_self_questioning(self):
        """Self-doubt patterns (wait, actually, on second thought) are removed."""
        text = "The answer is X. Wait, let me reconsider. Actually, X is right."
        pruned = prune_cot(text)
        assert "Wait" not in pruned
        assert "Actually" not in pruned
        assert "X" in pruned

    def test_preserves_conclusion(self):
        """The final answer / conclusion is preserved verbatim."""
        text = "Some preamble. Maybe I'm wrong. Conclusion: the answer is 7."
        pruned = prune_cot(text)
        assert "the answer is 7" in pruned

    def test_caps_at_max_tokens(self):
        """Output respects max_tokens (word-approximation OK)."""
        long_text = " ".join(["word"] * 1000)
        pruned = prune_cot(long_text, max_tokens=50)
        assert len(pruned.split()) <= 50

    def test_empty_input_returns_empty(self):
        assert prune_cot("") == ""

    def test_already_clean_text_unchanged(self):
        """Text without hedge patterns should pass through nearly unchanged."""
        text = "The answer is 42. Reasoning: 6 times 7 equals 42."
        pruned = prune_cot(text)
        assert "42" in pruned
        assert "6 times 7" in pruned


class TestIntegrationContract:
    """Both functions must be importable and have the documented signatures."""

    def test_shuffle_signature(self):
        import inspect

        sig = inspect.signature(shuffle_traces)
        assert "seed" in sig.parameters
        assert sig.parameters["seed"].default is None

    def test_prune_signature(self):
        import inspect

        sig = inspect.signature(prune_cot)
        assert "max_tokens" in sig.parameters
