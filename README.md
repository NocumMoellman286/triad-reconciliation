# Temporal Vision-Language Reconciliation of Optic Disc Images and Visual Fields

## Overview

Glaucoma follow-up rests on two readings that frequently disagree: the optic disc and the
visual field. This release implements the reconciliation stack the manuscript describes. Two
encoders read the structural and functional streams, a non-anticipative alignment operator
places both streams on the clinical timeline, a cross-stream attention pass builds a
reconciling state, and three readouts follow from it — a progression probability, an
arbitration direction over the three classes *structural*, *functional* and *concordant*, and
a stream-evidence weight. A hidden agreement state re-weights the objective toward the
discordant pairs, and an attribution-consistency term penalises a decision whose reported
weight contradicts it.

The package also carries the fourteen comparators the tables report against, the statistical
layer (patient-clustered bootstrap intervals, paired differences on shared units, a
fixed-sequence test over the declared hypotheses, a split-conformal deferral rule), the
stratified and workflow accounting, and a verification pass that checks the release against
the manuscript's own numbers.

**The analysed records are not distributed, and this release produces nothing in their
place.** The cohort is a private anonymised multicentre set whose institution and site
identities the manuscript withholds. Every command that reads records takes the table from the
caller with `--cohort` and stops without one, so no record is shipped, generated or inferred.
Consequently no record-level value in the manuscript is claimed here; each is reported
`BLOCKED` with the staging instruction in `verification_report.json`, and the numerical checks
run on tensor batches drawn from a recorded seed.

## What the artefacts hold

| file | what it is |
| --- | --- |
| `claim_to_code.json` | eighty manuscript claims, each with its location and the released symbol that carries it, plus the engineering defaults and the recorded deviations |
| `verification_report.json` | the two verification layers, the manuscript arithmetic, the configuration conformance, the hygiene scan and every blocked record-level check with its reason |
| `verification_summary.txt` | the same verdicts as a plain-text summary |
| `dataset_urls.txt` | the adjudicated data endpoints with their access conditions and what the probe returned |
| `integrity_manifest.json` | a SHA-256 digest per shipped file, written last |

## Installation

```bash
# pip, from the release root
python3 -m pip install -r requirements.txt -r requirements-dev.txt
export PYTHONPATH="$PWD/src"

# conda
conda env create -f environment.yml
conda activate triad-reconciliation

# container
docker build -t triad-reconciliation .
```

The package is not installed into the environment by default; `PYTHONPATH=src` is what the
scripts use and what `pyproject.toml` sets for the test runner.

## Data

The pipeline reads one JSONL cohort table discriminated by a `record` field: eye-level
covariates, structural visits, perimetry reports and labelled visit-pairs. The schema is
described by `triad_reconciliation.cohort.schema.schema_card()`. The release neither ships nor
generates a table, so a command cannot read records without one:

```bash
# check a table before a run reads it
bash scripts/prepare_data.sh --cohort /path/to/cohort.jsonl

# every command that reads records takes the same argument
python -m triad_reconciliation.cli.train --experiment configs/experiment/main.yaml \
    --cohort /path/to/cohort.jsonl
```

| source | role | access |
| --- | --- | --- |
| private anonymised multicentre cohort, four sites in two regions | development and held-out evaluation | not distributed and not reproduced here; see `dataset_urls.txt` |
| PAPILA | auxiliary fundus images with clinical data of both eyes | open, GPL 3.0+ |
| ACRIMA | auxiliary labelled fundus images | open for research use |
| RIM-ONE DL | auxiliary unified retinal image database | open for research and teaching |
| ODIR-5K | auxiliary multi-label ocular disease set | challenge terms |

The four auxiliary corpora carry 9,678 images in total, which is the number Sec. 5.2 prints
and the sum of the counts the sources publish. They are used for representation pretraining
only; nothing from them enters the reported tables and none of their images ship here.
`dataset_urls.txt` records what each probe returned.

## Configuration

Each experiment file declares the component files it builds on and then overrides the
sections it changes:

```yaml
defaults:
  - model/reconciliation.yaml
  - data/cohort.yaml
  - train/protocol.yaml
name: main
variant: full
model: {}
```

`configs/model/reconciliation.yaml` holds the stack shape, `configs/data/cohort.yaml` the
reading convention — the two clinical grids a visit-pair is taken between, the region held out
and the split the conformal threshold is fitted on — and `configs/train/protocol.yaml` the
optimisation settings beside the statistical design constants. No count, follow-up
distribution or label mix is fixed anywhere, because none of those is the reader's to choose.
`configs/experiment/_smoke.yaml` exists only to drive the unit tests and says so.

## Training

Algorithm 1 prints AdamW under a cosine schedule; the learning rate, weight decay, batch size,
warm-up, epoch count and precision are released values because the manuscript prints no
schedule. One command per reported configuration:

```bash
bash scripts/launch_train.sh configs/experiment/main.yaml /path/to/cohort.jsonl
bash scripts/launch_train.sh configs/experiment/ablation_without_reconciliation_state.yaml /path/to/cohort.jsonl
bash scripts/launch_train.sh configs/experiment/ablation_without_monotone_causal_alignment.yaml /path/to/cohort.jsonl
bash scripts/launch_train.sh configs/experiment/ablation_interaction_state_and_head.yaml /path/to/cohort.jsonl
# ... one per file under configs/experiment/
```

Reports and checkpoints land under `runs/train/`. A checkpoint carries the seed it was trained
under and a resumed run restores it.

## Evaluation

```bash
bash scripts/launch_eval.sh configs/experiment/main.yaml /path/to/cohort.jsonl
python -m triad_reconciliation.cli.tables --experiment configs/experiment/main.yaml \
    --cohort /path/to/cohort.jsonl
```

The first command writes the held-out discrimination, calibration and agreement numbers for
one configuration; the second renders the discrimination table across all fifteen roster rows,
the component analysis, the design analyses and the declared strata. Both write `.txt` reports
and a JSON companion under `runs/`.

The workflow and silent-mode tables are rendered with `---` against every row: no deployment
log ships with this release, and the manuscript's own values are held by the participating
sites. The same applies to every discrimination, calibration and agreement row until a cohort
table is staged.

## Verification

```bash
bash scripts/launch_verify.sh
bash scripts/launch_verify.sh   # run twice; the second pass is the one that can settle
```

Run it twice with no edits in between. The check that compares the shipped manifest with the
live tree can only read a manifest a previous run wrote, so the first pass records it as
`NOT_RUN` and the second settles it. The tool gates (`pytest`, `ruff`, `black`, `isort`,
`mypy`, `docker build`) run inside the shipping invocation; a caller that is already inside
pytest passes `include_gates=False`.

The shipped artefacts must be a fixed point: two consecutive runs with no edits in between
produce byte-identical `claim_to_code.json`, `verification_report.json`,
`verification_summary.txt` and `integrity_manifest.json`.

## Tests

```bash
python -m pytest -q
ruff check .
black --check .
isort --check-only .
mypy --strict src/triad_reconciliation
```

The suite covers the schema and the reader against the smallest table it accepts, the pairing
rule and the predictor embargo, the partitions and the declared strata, every loss term against
a hand-computed value, the discrimination, calibration, agreement and conformal metrics against
brute-force recomputations, the alignment operator's causality and monotonicity, the pooling
that keeps a padded row finite, the resampling and the fixed sequence, all fifteen comparators,
the reader-study and deployment accounting, the training schedule, a checkpoint round trip, a
single-batch overfit and a multi-batch training loop. Tests that need records skip rather than
manufacture them.

## Compute budget

The manuscript reports no hardware, wall-clock or storage figure, so this release records the
arithmetic instead of a measured budget. The reference model holds 367,066 parameters and the
largest tensor a batch carries is `batch × 8 visits × 29 tokens × 96`, which is under a
megabyte at batch 24. The reported budget of 32 epochs at 24 development pairs per step
therefore runs in minutes on a single accelerator and in tens of minutes on CPU. Training,
evaluation and every shipped check were exercised on CPU while this release was assembled; no
accelerator-only path is required and no distributed launcher is needed to reproduce the
shipped artefacts.

## Licensing

The release is MIT-licensed; see `LICENSE`. `THIRD_PARTY_NOTICES.txt` names every third-party
dataset and package with its real licence and source. No third-party data, and no record from
the analysed cohort, is shipped, generated or inferred here.
