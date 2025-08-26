#!/usr/bin/env bash
set -euo pipefail

# Simple runner to execute scheduler_client.py directly.
# Usage: ./examples/pm2/runner.sh [args passthrough]

PY_BIN="${PY_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${PY_BIN}" "${EXAMPLES_DIR}/scheduler_client.py" "$@"
