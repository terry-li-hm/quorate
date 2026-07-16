#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
local_bin="${HOME}/.local/bin"
effector="${HOME}/germline/effectors/quorate"

if [ ! -x "$effector" ]; then
  echo "Missing executable Quorate effector: $effector" >&2
  exit 1
fi

uv tool install --reinstall --no-cache --from "$repo" quorate
mkdir -p "$local_bin"
ln -sfn "$effector" "$local_bin/quorate"

"$local_bin/quorate" --version
