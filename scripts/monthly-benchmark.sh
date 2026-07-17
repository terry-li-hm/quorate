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

if ! /Users/terry/.local/bin/quorate-core usage --days 30 --json >/dev/null 2>&1; then
  print -u2 "quorate monthly usage snapshot failed"
fi
