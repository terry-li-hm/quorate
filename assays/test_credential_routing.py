"""Static regression checks for guarded Quorate credential routing."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_TEMPLATE = ROOT / "quorate.env.op"
MONTHLY_RUNNER = ROOT / "scripts/monthly-benchmark.sh"


def test_quorate_environment_uses_only_canonical_vault_references() -> None:
    text = ENV_TEMPLATE.read_text()

    assert "op://Agents/" not in text
    assert (
        "KIMI_CODE_API_KEY="
        "op://AgentSecrets/kimi-code-membership-api-key/credential"
    ) in text


def test_monthly_runner_checks_kimi_before_any_provider_call() -> None:
    text = MONTHLY_RUNNER.read_text()

    preflight = text.index('if [[ -z "${KIMI_CODE_API_KEY:-}" ]]')
    provider_call = text.index(
        "/Users/terry/.local/bin/quorate-core benchmark --json"
    )
    assert preflight < provider_call
