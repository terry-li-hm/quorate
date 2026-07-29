#!/bin/zsh
set -euo pipefail

export HOME=/Users/terry
export PATH=/Users/terry/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin

resolved_env=/Users/terry/.env.resolved
if [[ ! -f "$resolved_env" ]]; then
  print -u2 "quorate monthly benchmark: missing $resolved_env"
  exit 1
fi

set -a
source "$resolved_env"
set +a

fireworks_us_model=accounts/fireworks/routers/kimi-k3-us
fireworks_selected=0
if [[ "${CONSILIUM_MODEL_M1:-}" == "$fireworks_us_model" \
  || "${CONSILIUM_MODEL_M2:-}" == "$fireworks_us_model" \
  || "${CONSILIUM_MODEL_M3:-}" == "$fireworks_us_model" \
  || "${CONSILIUM_MODEL_M4:-}" == "$fireworks_us_model" \
  || "${CONSILIUM_MODEL_M5:-}" == "$fireworks_us_model" \
  || "${CONSILIUM_MODEL_M6:-}" == "$fireworks_us_model" ]]; then
  fireworks_selected=1
fi
if [[ "$fireworks_selected" -eq 1 \
  && -z "${QUORATE_FIREWORKS_KEY:-}" \
  && -z "${FIREWORKS_API_KEY:-}" ]]; then
  print -u2 "quorate monthly benchmark: Fireworks US trial is selected but its key is absent"
  exit 1
fi

output=$(mktemp "${TMPDIR:-/tmp}/quorate-monthly-benchmark.XXXXXX")
trap 'rm -f "$output"' EXIT

if ! /Users/terry/.local/bin/quorate-core benchmark --json >"$output" 2>&1; then
  cat "$output"
  exit 1
fi

status=$(
  /usr/bin/python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["result"]["status"])' \
    "$output"
)

if [[ "$status" != "healthy" ]]; then
  cat "$output"
fi

checker_args=("$output")
if [[ "$fireworks_selected" -eq 1 ]]; then
  checker_args=(--required "$output")
fi
if /usr/bin/python3 /Users/terry/code/quorate/scripts/check_fireworks_benchmark.py "${checker_args[@]}"; then
  :
else
  gate_status=$?
  if [[ "$gate_status" -eq 1 ]]; then
    cat "$output"
    print -u2 "quorate monthly benchmark: Fireworks US seat failed its exact-route gate"
    exit 1
  elif [[ "$gate_status" -ne 2 ]]; then
    print -u2 "quorate monthly benchmark: could not validate the Fireworks US route"
    exit 1
  fi
fi

if ! /Users/terry/.local/bin/quorate-core usage --days 30 --json >/dev/null 2>&1; then
  print -u2 "quorate monthly usage snapshot failed"
fi
