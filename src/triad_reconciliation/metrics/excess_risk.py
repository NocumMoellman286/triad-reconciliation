"""The excess-risk decomposition of the arbitration classifier.

The decomposition splits the excess risk of the arbitration readout into a term carried by
the temporal-alignment error and a term carried by the inverse square root of the effective
number of discordant training pairs. The second term is the one the re-weighting acts on: a
discordant pair that is up-weighted raises the effective count, and the effective count is
what sets the size of the term.

Ref: Theorem 1 and its corollary, Sec. 5.1 and Sec. 5.5; Table 3 reports the alignment error
in years beside the agreement the design predicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Bootstrap = np.ndarray


def alignment_error(learned_years: np.ndarray, consensus_years: np.ndarray) -> float:
    """Median absolute discrepancy between the learned alignment and its consensus."""

    left = np.asarray(learned_years, dtype=np.float64)
    right = np.asarray(consensus_years, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("the two alignment series must share a shape")
    if left.size == 0:
        return float("nan")
    return float(np.median(np.abs(left - right)))


def effective_pairs(weights: np.ndarray) -> float:
    """Kish effective count of a weighted pair set."""

    values = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return 0.0
    total = float(values.sum())
    squared = float((values**2).sum())
    if squared <= 0:
        return 0.0
    return total * total / squared


def pair_term(coefficient: float, effective: float) -> float:
    """The inverse-square-root contribution of the effective discordant count."""

    if effective <= 0:
        return float("inf")
    return float(coefficient) / float(np.sqrt(effective))


@dataclass(frozen=True)
class Decomposition:
    """A fitted two-term decomposition beside the measured excess it describes."""

    alignment_coefficient: float
    pair_coefficient: float
    alignment_component: float
    pair_component: float
    measured: float
    residual: float

    @property
    def predicted(self) -> float:
        return self.alignment_component + self.pair_component

    def as_mapping(self) -> dict[str, float]:
        return {
            "alignment_coefficient": self.alignment_coefficient,
            "pair_coefficient": self.pair_coefficient,
            "alignment_component": self.alignment_component,
            "pair_component": self.pair_component,
            "predicted": self.predicted,
            "measured": self.measured,
            "residual": self.residual,
        }


def fit_decomposition(
    alignment_errors: np.ndarray,
    effective_counts: np.ndarray,
    excess_risks: np.ndarray,
) -> tuple[float, float]:
    """Least-squares coefficients of the alignment term and the pair term."""

    errors = np.asarray(alignment_errors, dtype=np.float64)
    effective = np.asarray(effective_counts, dtype=np.float64)
    excess = np.asarray(excess_risks, dtype=np.float64)
    if not (errors.shape == effective.shape == excess.shape):
        raise ValueError("the three fit vectors must share a shape")
    if errors.size < 2:
        raise ValueError("a two-term fit needs at least two settings")
    design = np.stack((errors, 1.0 / np.sqrt(np.clip(effective, 1e-9, None))), axis=1)
    gram = design.T @ design + 1e-9 * np.eye(2)
    coefficients = np.linalg.solve(gram, design.T @ excess)
    return float(coefficients[0]), float(coefficients[1])


def decompose(
    alignment_errors: np.ndarray,
    effective_counts: np.ndarray,
    excess_risks: np.ndarray,
    measured: float,
) -> Decomposition:
    """Fit the two coefficients and report each component at the measured setting."""

    left, right = fit_decomposition(alignment_errors, effective_counts, excess_risks)
    alignment_component = left * float(np.asarray(alignment_errors, dtype=np.float64)[-1])
    pair_component = pair_term(right, float(np.asarray(effective_counts, dtype=np.float64)[-1]))
    return Decomposition(
        alignment_coefficient=left,
        pair_coefficient=right,
        alignment_component=alignment_component,
        pair_component=pair_component,
        measured=float(measured),
        residual=float(measured) - (alignment_component + pair_component),
    )


def scaling_ratio(small: float, large: float) -> float:
    """How much the pair term falls when the effective count grows from one to the other."""

    if small <= 0 or large <= 0:
        raise ValueError("effective counts must be positive")
    return float(np.sqrt(large / small))


def excess_risk(error_rates: np.ndarray, reference_rate: float) -> float:
    """Mean excess of a set of error rates over a reference rate."""

    values = np.asarray(error_rates, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(values.mean() - float(reference_rate))
