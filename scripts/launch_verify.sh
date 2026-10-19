#!/usr/bin/env bash
# Run the verification pass and write the released artefacts.
#
# Run it twice with no edits in between: the manifest-freshness check can only read a
# manifest that a previous run wrote.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

"${PYTHON_BIN}" -m triad_reconciliation.cli.verify
