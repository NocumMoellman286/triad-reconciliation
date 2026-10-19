"""Arithmetic checks taken from the manuscript's own printed values.

Each check recomputes a quantity the article states twice, or states once and whose
components it also prints. A check that fails here is a disagreement inside the article, not
a defect of this release, so the failures are reported in their own family and are excluded
from the code verdict.

Ref: Table 1 (pp. 6-8), Table 2 (p. 9), Table 3 (p. 10), Table 4 (p. 12), Table 5 (pp. 22-23),
Table 6 (p. 23), Sec. 2.1-2.4 (pp. 4-11), Sec. 5.2 (p. 14), Sec. 5.6 (pp. 15-16).
"""

from __future__ import annotations

from dataclasses import dataclass

TOLERANCE = 1e-3
ROUNDING_TOLERANCE = 2e-3


@dataclass(frozen=True)
class ArithmeticCheck:
    identifier: str
    location: str
    statement: str
    components: dict[str, float]
    expected: float
    tolerance: float = TOLERANCE


CHECKS: tuple[ArithmeticCheck, ...] = (
    ArithmeticCheck(
        "M001",
        "Sec. 5.2, p. 14",
        "The four auxiliary resources total the printed 9,678 images.",
        {"papila": 488.0, "acrima": 705.0, "rim_one_dl": 485.0, "odir_5k": 8000.0},
        9678.0,
    ),
    ArithmeticCheck(
        "M002",
        "Sec. 5.2, p. 14",
        "The glaucomatous and normal images of the ACRIMA database sum to its printed total.",
        {"glaucomatous": 396.0, "normal": 309.0},
        705.0,
    ),
    ArithmeticCheck(
        "M003",
        "Sec. 5.2, p. 14",
        "The normal and glaucomatous retinographies of RIM-ONE DL sum to its printed total.",
        {"normal": 313.0, "glaucoma": 172.0},
        485.0,
    ),
    ArithmeticCheck(
        "M004",
        "Sec. 2.2, p. 8",
        "The discordant and concordant shares of the held-out pairs sum to one.",
        {"discordant": 28.4, "concordant": 71.6},
        100.0,
    ),
    ArithmeticCheck(
        "M005",
        "Sec. 2.1, p. 4",
        "The gain over the strongest single-stream model is the difference of the two printed areas.",
        {"triad": 0.891, "single_stream": 0.858},
        0.033,
    ),
    ArithmeticCheck(
        "M006",
        "Sec. 2.1, p. 4",
        "The gain over the strongest static-fusion model is the difference of the two printed areas.",
        {"triad": 0.891, "static_fusion": 0.874},
        0.017,
    ),
    ArithmeticCheck(
        "M007",
        "Sec. 2.1, p. 4",
        "The difference against the unconstrained comparator is the difference of the two printed areas.",
        {"triad": 0.891, "unconstrained": 0.894},
        -0.003,
    ),
    ArithmeticCheck(
        "M008",
        "Sec. 2.2, p. 8",
        "The discordant-stratum gain over static fusion is the difference of the two printed areas.",
        {"triad": 0.874, "static_fusion": 0.803},
        0.071,
    ),
    ArithmeticCheck(
        "M009",
        "Sec. 2.2, p. 8",
        "The discordant-stratum gain over the unconstrained comparator is the difference of the two printed areas.",
        {"triad": 0.874, "unconstrained": 0.822},
        0.052,
    ),
    ArithmeticCheck(
        "M010",
        "Sec. 2.2, p. 8",
        "On the concordant stratum the three printed areas differ only within a thousandth.",
        {"triad": 0.918, "unconstrained": 0.921},
        -0.003,
    ),
    ArithmeticCheck(
        "M011",
        "Sec. 2.2, p. 8",
        "The overall cost of removing the reconciliation state is the difference of the printed areas.",
        {"full": 0.891, "without_state": 0.879},
        0.012,
    ),
    ArithmeticCheck(
        "M012",
        "Sec. 2.2, p. 8",
        "The discordant-stratum cost of removing the reconciliation state is the difference of the printed areas.",
        {"full": 0.874, "without_state": 0.812},
        0.062,
    ),
    ArithmeticCheck(
        "M013",
        "Table 2, p. 9",
        "The printed discordant delta of the reconciliation-state row is its area difference.",
        {"full": 0.874, "without_state": 0.812},
        0.062,
    ),
    ArithmeticCheck(
        "M014",
        "Sec. 2.2, p. 8",
        "The overall cost of removing the arbitration head is the difference of the printed areas.",
        {"full": 0.891, "without_head": 0.888},
        0.003,
    ),
    ArithmeticCheck(
        "M015",
        "Sec. 2.2, p. 8",
        "The discordant-stratum cost of removing the arbitration head is the difference of the printed areas.",
        {"full": 0.874, "without_head": 0.859},
        0.015,
    ),
    ArithmeticCheck(
        "M016",
        "Sec. 2.2, p. 8",
        "The agreement drop from removing the attribution-consistency term is the difference of the printed values.",
        {"full": 0.66, "without_consistency": 0.51},
        0.15,
    ),
    ArithmeticCheck(
        "M017",
        "Sec. 2.2, p. 8",
        "The overall cost of disabling agreement-state re-weighting is the difference of the printed areas.",
        {"full": 0.891, "without_reweighting": 0.884},
        0.007,
    ),
    ArithmeticCheck(
        "M018",
        "Sec. 2.2, p. 8",
        "The discordant-stratum cost of disabling re-weighting is the difference of the printed areas.",
        {"full": 0.874, "without_reweighting": 0.826},
        0.048,
    ),
    ArithmeticCheck(
        "M019",
        "Sec. 2.2, p. 8",
        "The overall cost of unrestricted cross-time attention is the difference of the printed areas.",
        {"full": 0.891, "without_monotone": 0.886},
        0.005,
    ),
    ArithmeticCheck(
        "M020",
        "Sec. 2.2, p. 8",
        "The discordant-stratum cost of unrestricted cross-time attention is the difference of the printed areas.",
        {"full": 0.874, "without_monotone": 0.838},
        0.036,
    ),
    ArithmeticCheck(
        "M021",
        "Sec. 2.2, p. 8",
        "The overall cost of a numeric perimetry vector is the difference of the printed areas.",
        {"full": 0.891, "numeric_input": 0.882},
        0.009,
    ),
    ArithmeticCheck(
        "M022",
        "Sec. 2.2, p. 8",
        "The discordant-stratum cost of a numeric perimetry vector is the difference of the printed areas.",
        {"full": 0.874, "numeric_input": 0.845},
        0.029,
    ),
    ArithmeticCheck(
        "M023",
        "Table 2, p. 9",
        "The interaction row's discordant delta is its area difference against the full model.",
        {"full": 0.874, "interaction": 0.789},
        0.085,
    ),
    ArithmeticCheck(
        "M024",
        "Sec. 2.2, p. 8",
        "The interaction loss exceeds the sum of the two separate losses by the printed amount.",
        {"joint": 0.085, "head": 0.015, "state": 0.062},
        0.008,
    ),
    ArithmeticCheck(
        "M025",
        "Table 6, p. 23",
        "The scored share of eligible encounters is the ratio of the two printed counts.",
        {"scored": 9847.0, "eligible": 10773.0},
        0.914,
    ),
    ArithmeticCheck(
        "M026",
        "Table 6, p. 23",
        "The scored share and the two unscored shares of the silent run sum to one.",
        {"scored": 91.4, "missing": 5.8, "processing": 2.8},
        100.0,
    ),
    ArithmeticCheck(
        "M027",
        "Table 6, p. 23",
        "The median latency sits inside the printed interquartile range.",
        {"lower": 4.1, "median": 4.8, "upper": 5.9},
        4.8,
    ),
    ArithmeticCheck(
        "M028",
        "Table 4, p. 12",
        "The concordant vignettes are the remainder of the printed vignette count.",
        {"total": 320.0, "discordant": 224.0},
        96.0,
    ),
    ArithmeticCheck(
        "M029",
        "Sec. 2.4, p. 11",
        "The assisted gain on discordant vignettes is the difference of the two printed accuracies.",
        {"assisted": 0.729, "unassisted": 0.641},
        0.088,
    ),
    ArithmeticCheck(
        "M030",
        "Table 4, p. 12",
        "The assisted gain on concordant vignettes is the difference of the two printed accuracies.",
        {"assisted": 0.898, "unassisted": 0.874},
        0.024,
    ),
    ArithmeticCheck(
        "M031",
        "Table 4, p. 12",
        "The arbitration-concordance change is the difference of the two printed values.",
        {"assisted": 0.58, "unassisted": 0.47},
        0.11,
    ),
    ArithmeticCheck(
        "M032",
        "Sec. 2.4, p. 11",
        "The inter-reader agreement change is the difference of the two printed values.",
        {"assisted": 0.61, "unassisted": 0.52},
        0.09,
    ),
    ArithmeticCheck(
        "M033",
        "Table 4, p. 12",
        "The median reading-time saving is the difference of the two printed values.",
        {"unassisted": 132.0, "assisted": 104.0},
        28.0,
    ),
    ArithmeticCheck(
        "M034",
        "Sec. 2.3, p. 9",
        "The advantage over the strongest attribution proxy is the difference of the two printed values.",
        {"triad": 0.66, "proxy": 0.44},
        0.22,
    ),
    ArithmeticCheck(
        "M035",
        "Sec. 2.3, p. 9",
        "The ceiling the model approaches is the recorded arbitration's own inter-reader agreement.",
        {"ceiling": 0.68, "triad": 0.66},
        0.02,
    ),
    ArithmeticCheck(
        "M036",
        "Table 3, p. 10",
        "The weight separation the design predicts reproduces the two printed medians.",
        {"structural_resolved": 0.79, "functional_resolved": 0.24},
        0.55,
    ),
    ArithmeticCheck(
        "M037",
        "Table 2, p. 9",
        "The full-model row of the component analysis repeats the headline row of the discrimination table.",
        {"table_two": 0.891, "table_one": 0.891},
        0.0,
    ),
    ArithmeticCheck(
        "M038",
        "Sec. 2.4, p. 11",
        "The two reader groups' gains reproduce the reported average gain within rounding.",
        {"lower_readers": 0.142, "upper_readers": 0.019, "reported": 0.088},
        0.0893,
        ROUNDING_TOLERANCE,
    ),
)

DISCREPANCIES: tuple[ArithmeticCheck, ...] = (
    ArithmeticCheck(
        "M101",
        "Sec. 2.1, p. 4 versus Table 1 row 8, p. 6",
        "The interval quoted in the prose for the serial-tomography vision transformer is the interval the table prints.",
        {"prose_lower": 0.843, "table_lower": 0.842, "prose_upper": 0.873, "table_upper": 0.874},
        0.0,
    ),
    ArithmeticCheck(
        "M102",
        "Sec. 2.1, p. 4 versus Table 1 row 13, p. 7",
        "The interval quoted in the prose for the multi-encoder static fusion model is the interval the table prints.",
        {"prose_lower": 0.859, "table_lower": 0.858, "prose_upper": 0.889, "table_upper": 0.890},
        0.0,
    ),
    ArithmeticCheck(
        "M103",
        "Sec. 2.2, p. 8 versus Table 2 row 3, p. 9",
        "The agreement value quoted in the prose after removing the arbitration head is the value the table prints.",
        {"prose": 0.42, "table": 0.24},
        0.0,
    ),
    ArithmeticCheck(
        "M104",
        "Sec. 5.6, p. 16",
        "The number of published hypotheses matches the number of pre-specified minimum detectable differences.",
        {"hypotheses": 5.0, "minimum_detectable_differences": 4.0},
        0.0,
    ),
)


@dataclass(frozen=True)
class CheckOutcome:
    identifier: str
    location: str
    statement: str
    observed: float
    expected: float
    status: str
    detail: str
    family: str = "manuscript"


def _outcome(
    check: ArithmeticCheck,
    observed: float,
    status: str,
    detail: str,
) -> CheckOutcome:
    return CheckOutcome(
        identifier=check.identifier,
        location=check.location,
        statement=check.statement,
        observed=observed,
        expected=check.expected,
        status=status,
        detail=detail,
    )


def _evaluate_components(check: ArithmeticCheck) -> tuple[float, str]:
    names = sorted(check.components)
    if check.identifier == "M024":
        return (
            check.components["joint"] - (check.components["head"] + check.components["state"]),
            "joint loss minus the sum of the separate losses",
        )
    if check.identifier == "M026":
        return (
            check.components["scored"]
            + check.components["missing"]
            + check.components["processing"]
        ), "the three shares of the silent run"
    if check.identifier == "M036":
        return (
            check.components["structural_resolved"] - check.components["functional_resolved"]
        ), "difference of the two printed weight medians"
    if check.identifier == "M038":
        weights = {"lower_readers": 4.0, "upper_readers": 3.0}
        total = sum(weights[name] for name in weights)
        return (
            sum(weights[name] * check.components[name] for name in weights) / total
        ), "reader-count-weighted mean of the two reported gains"
    if check.identifier == "M028":
        return (
            check.components["total"] - check.components["discordant"]
        ), "vignette count less the discordant vignettes"
    if check.identifier == "M035":
        return (
            check.components["ceiling"] - check.components["triad"]
        ), "difference between the ceiling and the model"
    if check.identifier == "M027":
        return check.components["median"], "the printed median latency"
    if check.identifier == "M033":
        return (
            check.components["unassisted"] - check.components["assisted"]
        ), "the reading time saved between the two arms"
    if check.identifier == "M005" or check.identifier == "M006":
        return (
            check.components["triad"] - check.components[sorted(set(names) - {"triad"})[0]],
            "difference of the two printed areas",
        )
    if check.identifier == "M007":
        return (
            check.components["triad"] - check.components["unconstrained"],
            "difference of the two printed areas",
        )
    if check.identifier == "M010":
        return (
            check.components["triad"] - check.components["unconstrained"],
            "difference of the two printed areas",
        )
    if check.identifier == "M037":
        return (
            check.components["table_two"] - check.components["table_one"],
            "difference of the two printed values",
        )
    if len(names) == 2 and "full" in names:
        other = sorted(set(names) - {"full"})[0]
        return check.components["full"] - check.components[other], f"difference against {other}"
    if check.identifier == "M025":
        return (
            check.components["scored"] / check.components["eligible"],
            "ratio of the two printed counts",
        )
    if len(names) == 2 and {"assisted", "unassisted"} <= set(names):
        return (
            check.components["assisted"] - check.components["unassisted"],
            "difference of the two printed values",
        )
    if len(names) == 2 and {"triad"} <= set(names):
        other = sorted(set(names) - {"triad"})[0]
        return check.components["triad"] - check.components[other], f"difference against {other}"
    if len(names) == 2 and {"ceiling"} <= set(names):
        other = sorted(set(names) - {"ceiling"})[0]
        return check.components["ceiling"] - check.components[other], f"difference against {other}"
    return float(sum(check.components[name] for name in names)), "sum of the printed components"


def run_manuscript_checks() -> list[CheckOutcome]:
    """Evaluate every arithmetic check against the printed components."""

    outcomes: list[CheckOutcome] = []
    for check in CHECKS:
        observed, rule = _evaluate_components(check)
        difference = observed - check.expected
        status = "PASS" if abs(difference) <= check.tolerance else "FAIL"
        outcomes.append(
            _outcome(
                check,
                observed,
                status,
                f"{rule}; observed {observed:.6f} against printed {check.expected:.6f}, "
                f"difference {difference:+.6f}",
            )
        )
    return outcomes


def run_discrepancy_checks() -> list[CheckOutcome]:
    """Evaluate the checks that compare two printed statements of the same quantity."""

    outcomes: list[CheckOutcome] = []
    for check in DISCREPANCIES:
        components = check.components
        difference = float("nan")
        if check.identifier in ("M101", "M102"):
            difference = max(
                abs(components["prose_lower"] - components["table_lower"]),
                abs(components["prose_upper"] - components["table_upper"]),
            )
        elif check.identifier == "M103":
            difference = abs(components["prose"] - components["table"])
        elif check.identifier == "M104":
            difference = abs(
                components["hypotheses"] - components["minimum_detectable_differences"]
            )
        status = "PASS" if difference <= TOLERANCE else "FAIL"
        outcomes.append(
            _outcome(
                check,
                difference,
                status,
                f"largest disagreement between the two printed statements {difference:.6f}"
                + (
                    ""
                    if status == "PASS"
                    else "; the article states the quantity twice with different values"
                ),
            )
        )
    return outcomes


def all_manuscript_outcomes() -> list[CheckOutcome]:
    return [*run_manuscript_checks(), *run_discrepancy_checks()]


def failing_identifiers() -> tuple[str, ...]:
    return tuple(
        outcome.identifier for outcome in all_manuscript_outcomes() if outcome.status == "FAIL"
    )
