#!/usr/bin/env bash
# Check the cohort table a caller supplies before a run reads it.
#
# The analysed records are not distributed: the Data availability statement withholds both
# institution and site identities. This release neither ships nor generates them, so the
# command takes the table as an argument and stops without one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${1:-}" != "--cohort" ]]; then
  echo "usage: prepare_data.sh --cohort <table.jsonl>" >&2
  echo "The release does not distribute or generate the analysed records." >&2
  exit 2
fi
COHORT="${2:?usage: prepare_data.sh --cohort <table.jsonl>}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

"${PYTHON_BIN}" -c '
import sys
from pathlib import Path

from triad_reconciliation.cohort.reader import read_cohort
from triad_reconciliation.cohort.schema import schema_card
from triad_reconciliation.cohort.splits import coverage_report

table = read_cohort(Path(sys.argv[1]))
card = schema_card()
print("table      :", table.source)
print("visit-pairs:", table.n_pairs)
print("schema     :", card["format"], card["record_kinds"])
covered = coverage_report(table.pairs)
print("coverage   :", round(covered["coverage"], 4), "of the pairs carry both streams")
' "${COHORT}"
