#!/usr/bin/env bash
# Fit the reconciliation model on the development region.
#
# The package is not pip-installed in this tree, so the script puts its own src directory on
# the path and takes the interpreter from PYTHON_BIN.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT="${1:-configs/experiment/main.yaml}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

"${PYTHON_BIN}" -m triad_reconciliation.cli.train --experiment "${EXPERIMENT}" "$@"
