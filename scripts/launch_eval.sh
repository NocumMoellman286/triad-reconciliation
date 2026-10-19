#!/usr/bin/env bash
# Evaluate the fitted model on the held-out region and render the reported tables.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT="${1:-configs/experiment/main.yaml}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

"${PYTHON_BIN}" -m triad_reconciliation.cli.evaluate --experiment "${EXPERIMENT}"
"${PYTHON_BIN}" -m triad_reconciliation.cli.tables --experiment "${EXPERIMENT}"
