"""Temporal vision-language reconciliation of optic disc images and visual fields.

The package implements the reconciliation stack: a non-anticipative alignment operator over
two clinical timelines, a cross-stream reconciliation state, a three-class arbitration head
and a stream-evidence weight, together with the cohort construction, comparator roster and
statistical layer the study reports.

Ref: Sec. 5.1-5.8, Algorithm 1.
"""

__all__ = ["__version__"]

__version__ = "0.4.2"
