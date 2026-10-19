"""The paper-claim to code map.

Every claim names the manuscript location it was read from and the released symbol that
carries it. A claim resolves only when its symbols import and expose the named attribute, so
a rename breaks the map rather than passing silently.

Ref: the whole manuscript; the mapping is layer one of the verification pass.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

VerificationKind = Literal["mapping", "execution"]
Family = Literal["code", "manuscript"]


@dataclass(frozen=True)
class Claim:
    identifier: str
    location: str
    statement: str
    symbols: tuple[str, ...]
    verification: VerificationKind
    family: Family = "code"


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "C001",
        "Abstract, p. 1",
        "The study reconciles serial optic-disc imaging with serial perimetry rather than optimising an average discrimination.",
        ("triad_reconciliation.analysis.pipeline:run_study",),
        "mapping",
    ),
    Claim(
        "C002",
        "Abstract, p. 1",
        "The model reports, for each decision, how much of it rests on each stream.",
        (
            "triad_reconciliation.models.triad:TriadModel",
            "triad_reconciliation.models.reconciliation:CrossStreamReconciliation",
        ),
        "mapping",
    ),
    Claim(
        "C003",
        "Sec. 1, p. 3",
        "Reconciliation is expressed as a stream-evidence weight statement and a direction of arbitration.",
        ("triad_reconciliation.models.triad:TriadOutput",),
        "mapping",
    ),
    Claim(
        "C004",
        "Sec. 1, p. 3",
        "The consistency term prevents a contradiction between the expressed attribution and the decision.",
        ("triad_reconciliation.losses.consistency:consistency_loss",),
        "execution",
    ),
    Claim(
        "C005",
        "Sec. 1, p. 3",
        "A hidden agreement state modifies the objective for conflicting pairs.",
        ("triad_reconciliation.losses.reweighting:agreement_weights",),
        "execution",
    ),
    Claim(
        "C006",
        "Sec. 1, p. 3 (Proposition 1)",
        "The stream-evidence weight is identifiable only when the two streams' evidence is separated by more than the inverse Lipschitz constant of the fusion map.",
        (
            "triad_reconciliation.metrics.identifiability:identifiable",
            "triad_reconciliation.metrics.identifiability:estimate_lipschitz",
        ),
        "execution",
    ),
    Claim(
        "C007",
        "Sec. 1, p. 3 (Theorem 1)",
        "The excess risk of the arbitration classifier splits into a temporal-alignment term and a term in the effective number of discordant training pairs.",
        ("triad_reconciliation.metrics.excess_risk:decompose",),
        "execution",
    ),
    Claim(
        "C008",
        "Sec. 1, p. 3",
        "Fusion suffices for the concordant majority and reconciliation is needed only for the discordant minority.",
        ("triad_reconciliation.strata.partitions:discordance_strata",),
        "mapping",
    ),
    Claim(
        "C009",
        "Sec. 2.1, p. 4",
        "The two regions are separated, with testing on the sites that took no part in development.",
        ("triad_reconciliation.cohort.splits:split_regions",),
        "execution",
    ),
    Claim(
        "C010",
        "Sec. 2.1, p. 4",
        "Discrimination is reported as the area under the receiver-operating-characteristic curve.",
        ("triad_reconciliation.metrics.discrimination:auroc",),
        "execution",
    ),
    Claim(
        "C011",
        "Sec. 2.1, p. 4",
        "Precision-recall is reported as the area under the precision-recall curve.",
        ("triad_reconciliation.metrics.discrimination:average_precision",),
        "execution",
    ),
    Claim(
        "C012",
        "Sec. 2.1, p. 4",
        "Sensitivity is reported at a matched specificity.",
        ("triad_reconciliation.metrics.discrimination:sensitivity_at_specificity",),
        "execution",
    ),
    Claim(
        "C013",
        "Sec. 2.1, p. 5",
        "An alignment operator over the visit times is monotone and independent of any visit later than the query time.",
        ("triad_reconciliation.models.alignment:MonotoneCausalAlignment",),
        "execution",
    ),
    Claim(
        "C014",
        "Sec. 2.1, p. 5",
        "Serial optic-disc recordings feed a backbone adapted to optic-disc imaging.",
        ("triad_reconciliation.models.encoders:VisionEncoder",),
        "execution",
    ),
    Claim(
        "C015",
        "Sec. 2.1, p. 5",
        "Serial perimetry reports feed an adapter-enhanced language encoder that tokenises both streams into a common token space.",
        (
            "triad_reconciliation.models.encoders:LanguageEncoder",
            "triad_reconciliation.models.adapters:BottleneckAdapter",
        ),
        "execution",
    ),
    Claim(
        "C016",
        "Sec. 2.1, p. 5",
        "The reconciliation emits a calibrated probability, an arbitral direction, and the stream weight with its consistency term.",
        ("triad_reconciliation.metrics.conformal:deferral_analysis",),
        "execution",
    ),
    Claim(
        "C017",
        "Table 1, p. 6-8",
        "Fifteen methods are evaluated on the same test units, and each row carries discrimination, calibration and agreement.",
        ("triad_reconciliation.comparators.roster:ROSTER",),
        "mapping",
    ),
    Claim(
        "C018",
        "Table 1 caption, p. 6",
        "Agreement is Cohen's kappa between the predicted arbitration direction and the recorded arbitration, computed on the discordant subset.",
        ("triad_reconciliation.metrics.agreement:cohen_kappa",),
        "execution",
    ),
    Claim(
        "C019",
        "Table 1 caption, p. 6",
        "The three clinical rules emit a decision and no probability, so no calibration error is defined for them.",
        ("triad_reconciliation.comparators.clinical_rules:binary_decisions",),
        "execution",
    ),
    Claim(
        "C020",
        "Sec. 2.2, p. 8",
        "The advantage is confined to the visit-pairs whose streams disagree.",
        ("triad_reconciliation.analysis.pipeline:discordance_analysis",),
        "execution",
    ),
    Claim(
        "C021",
        "Sec. 2.2, p. 8",
        "The component analysis isolates the effect of removing the reconciliation state.",
        ("triad_reconciliation.models.triad:VariantFlags",),
        "execution",
    ),
    Claim(
        "C022",
        "Sec. 2.2, p. 8",
        "Removing the attribution-consistency term leaves discrimination roughly unchanged and reduces agreement.",
        ("triad_reconciliation.losses.consistency:contradiction_mask",),
        "execution",
    ),
    Claim(
        "C023",
        "Sec. 2.2, p. 8",
        "Removing the agreement-state re-weighting costs more in the discordant stratum than overall.",
        ("triad_reconciliation.losses.reweighting:discordant_weight_mass",),
        "execution",
    ),
    Claim(
        "C024",
        "Table 2, p. 9",
        "The interaction row removes the arbitration head and the reconciliation state together.",
        ("triad_reconciliation.models.triad:VariantFlags",),
        "mapping",
    ),
    Claim(
        "C025",
        "Sec. 2.3, p. 9",
        "Agreement is reported against the recorded reference arbitration, whose own inter-reader agreement is the attainable ceiling.",
        ("triad_reconciliation.readers.study:inter_reader_agreement",),
        "execution",
    ),
    Claim(
        "C026",
        "Sec. 2.3, p. 9",
        "The strongest attribution proxy is the post-hoc attention of the unconstrained comparator.",
        (
            "triad_reconciliation.comparators.timeline:ConcatenatedTimelineTransformer.attention_attribution",
        ),
        "execution",
    ),
    Claim(
        "C027",
        "Sec. 2.3, p. 9",
        "The design analyses report an alignment error in years beside the separation of the stream weight.",
        (
            "triad_reconciliation.metrics.excess_risk:alignment_error",
            "triad_reconciliation.metrics.identifiability:separation_of_weight",
        ),
        "execution",
    ),
    Claim(
        "C028",
        "Sec. 2.4, p. 11",
        "Machine advice is assessed in a three-arm reader study with a washout between arms.",
        ("triad_reconciliation.readers.study:summarise_arm",),
        "mapping",
    ),
    Claim(
        "C029",
        "Sec. 2.4, p. 11",
        "Over-reliance is the share of cases with wrong advice that the reader adopted.",
        ("triad_reconciliation.readers.reliance:reliance_report",),
        "execution",
    ),
    Claim(
        "C030",
        "Sec. 2.4, p. 11",
        "Under-reliance is the share of cases with correct advice that the reader rejected.",
        ("triad_reconciliation.readers.reliance:reliance_report",),
        "execution",
    ),
    Claim(
        "C031",
        "Sec. 2.4, p. 11",
        "Inter-reader agreement is computed over the reader-by-vignette matrix.",
        ("triad_reconciliation.readers.study:reader_matrix",),
        "execution",
    ),
    Claim(
        "C032",
        "Sec. 3.1, p. 11",
        "The reliance asymmetry, not the accuracy of the model, limits the benefit of the advice.",
        ("triad_reconciliation.readers.reliance:RelianceReport.asymmetry",),
        "execution",
    ),
    Claim(
        "C033",
        "Sec. 5.1, p. 13",
        "The structural stream gathers optic-disc imaging and the functional stream gathers perimetry reports.",
        (
            "triad_reconciliation.core.records:StructuralVisit",
            "triad_reconciliation.core.records:PerimetryVisit",
        ),
        "mapping",
    ),
    Claim(
        "C034",
        "Sec. 5.1, p. 13",
        "A visit-pair is an index visit with its closest neighbour on the other timeline.",
        ("triad_reconciliation.cohort.visit_pairs:build_visit_pairs",),
        "execution",
    ),
    Claim(
        "C035",
        "Sec. 5.1, p. 13",
        "The arbitration label takes the values structural, functional and concordant.",
        ("triad_reconciliation.core.records:ARBITRATION_CLASSES",),
        "mapping",
    ),
    Claim(
        "C036",
        "Sec. 5.1, p. 13",
        "The model maps the two streams to a progression probability, an arbitration direction and a stream weight.",
        ("triad_reconciliation.models.triad:TriadModel.forward",),
        "execution",
    ),
    Claim(
        "C037",
        "Sec. 5.1, p. 13",
        "Two estimands are specified beforehand: discrimination for progression and kappa of the direction on discordant pairs.",
        ("triad_reconciliation.inference.sequential:fixed_sequence",),
        "execution",
    ),
    Claim(
        "C038",
        "Sec. 5.2, p. 13",
        "The cohort is a private anonymised multicentre set at four sites in two regions, and the release distributes no record from it.",
        (
            "triad_reconciliation.cohort.reader:read_cohort",
            "triad_reconciliation.cohort.schema:schema_card",
        ),
        "mapping",
    ),
    Claim(
        "C039",
        "Sec. 5.2, p. 14",
        "A strict patient-level separation is held between development and testing.",
        ("triad_reconciliation.cohort.splits:Partition.overlap",),
        "execution",
    ),
    Claim(
        "C040",
        "Sec. 5.2, p. 14",
        "The units of analysis are the patient, the eye, the visit and the visit-pair; the released reader reads the table the caller supplies and the schema declares what it must carry.",
        (
            "triad_reconciliation.cohort.reader:read_cohort",
            "triad_reconciliation.cohort.splits:coverage_report",
        ),
        "execution",
    ),
    Claim(
        "C041",
        "Sec. 5.2, p. 14",
        "Four auxiliary resources are used for representation pretraining, and each has an access route, conditions and an image count.",
        ("triad_reconciliation.comparators.roster:ROSTER",),
        "mapping",
    ),
    Claim(
        "C042",
        "Sec. 5.3, p. 14",
        "The reference standard is agreement between three specialists with adjudication by a fourth.",
        ("triad_reconciliation.readers.study:Vignette",),
        "mapping",
    ),
    Claim(
        "C043",
        "Sec. 5.3, p. 14",
        "Reference arbitration is structural-dominant, functional-dominant or unresolved at the review.",
        ("triad_reconciliation.core.records:ArbitrationLabel",),
        "mapping",
    ),
    Claim(
        "C044",
        "Sec. 5.3, p. 14",
        "Two further progression standards are analysed independently for sensitivity of the conclusions.",
        ("triad_reconciliation.comparators.clinical_rules:all_rules",),
        "execution",
    ),
    Claim(
        "C045",
        "Sec. 5.4, p. 14",
        "The alignment weights increase along the clinical timeline and are exactly zero for any source at or beyond the query time.",
        ("triad_reconciliation.models.alignment:MonotoneCausalAlignment.forward",),
        "execution",
    ),
    Claim(
        "C046",
        "Sec. 5.4, p. 15",
        "The reconciling state is obtained through cross-stream attention and the stream weight is its sigmoid readout.",
        ("triad_reconciliation.models.reconciliation:CrossStreamReconciliation.forward",),
        "execution",
    ),
    Claim(
        "C047",
        "Sec. 5.5, p. 15",
        "A softmax classifier over the three arbitration classes translates the reconciling state into the direction.",
        ("triad_reconciliation.models.heads:ArbitrationHead",),
        "execution",
    ),
    Claim(
        "C048",
        "Sec. 5.5, p. 15",
        "The agreement state is inferred from the same latent as the direction.",
        ("triad_reconciliation.models.heads:AgreementHead",),
        "execution",
    ),
    Claim(
        "C049",
        "Sec. 5.5, p. 15",
        "The pair weight is one plus the re-weighting strength on a pair read as discordant.",
        ("triad_reconciliation.losses.reweighting:agreement_weights",),
        "execution",
    ),
    Claim(
        "C050",
        "Sec. 5.5, p. 15",
        "The consistency term penalises a decision whose attribution contradicts it, by the distance to the margin.",
        ("triad_reconciliation.losses.consistency:consistency_loss",),
        "execution",
    ),
    Claim(
        "C051",
        "Sec. 5.5, p. 15",
        "The objective is the weighted sum of the progression, arbitration and consistency terms over the summed weights.",
        ("triad_reconciliation.losses.objective:reconciliation_objective",),
        "execution",
    ),
    Claim(
        "C052",
        "Algorithm 1, p. 16",
        "Training initialises the vision encoder, the adapter-augmented language encoder, the reconciliation module and the arbitration head.",
        ("triad_reconciliation.models.triad:TriadModel.__init__",),
        "execution",
    ),
    Claim(
        "C053",
        "Algorithm 1, p. 16",
        "The structural tokens carry an acquisition-time encoding and the functional tokens a visit-time encoding.",
        ("triad_reconciliation.streams.timeenc:sinusoidal_time",),
        "execution",
    ),
    Claim(
        "C054",
        "Algorithm 1, p. 16",
        "The parameters are updated with AdamW under a cosine schedule.",
        (
            "triad_reconciliation.analysis.trainer:build_optimizer",
            "triad_reconciliation.analysis.trainer:learning_rate_at",
        ),
        "execution",
    ),
    Claim(
        "C055",
        "Algorithm 1, p. 16",
        "The epoch loop visits each mini-batch of development visit-pairs and updates on the weighted objective.",
        ("triad_reconciliation.analysis.trainer:fit_binary_module",),
        "execution",
    ),
    Claim(
        "C056",
        "Sec. 5.6, p. 15",
        "Discrimination is reported separately as area under the curve and as sensitivity at a fixed specificity.",
        ("triad_reconciliation.metrics.discrimination:roc_curve",),
        "execution",
    ),
    Claim(
        "C057",
        "Sec. 5.6, p. 15",
        "Calibration is reported as expected calibration error, Brier score, calibration slope and intercept.",
        (
            "triad_reconciliation.metrics.calibration:calibration_slope_intercept",
            "triad_reconciliation.metrics.calibration:expected_calibration_error",
        ),
        "execution",
    ),
    Claim(
        "C058",
        "Sec. 5.6, p. 15",
        "Comparative intervals are cluster-bootstrapped with the patient as the resampling unit.",
        ("triad_reconciliation.inference.cluster_bootstrap:bootstrap_interval",),
        "execution",
    ),
    Claim(
        "C059",
        "Sec. 5.6, p. 16",
        "Kappa is reported with a cluster-bootstrap interval.",
        ("triad_reconciliation.inference.cluster_bootstrap:percentile_interval",),
        "execution",
    ),
    Claim(
        "C060",
        "Sec. 5.6, p. 16",
        "Multiplicity is controlled by a fixed-sequence method in the stated order.",
        ("triad_reconciliation.inference.sequential:fixed_sequence",),
        "execution",
    ),
    Claim(
        "C061",
        "Sec. 5.6, p. 16",
        "Each comparison runs in matched pairs on the same test units so no cohort is blended.",
        ("triad_reconciliation.inference.paired:paired_difference",),
        "execution",
    ),
    Claim(
        "C062",
        "Sec. 5.6, p. 16",
        "Two thousand bootstrap replicates are taken and a percentile interval is evaluated.",
        ("triad_reconciliation.core.config:InferenceConfig",),
        "mapping",
    ),
    Claim(
        "C063",
        "Sec. 5.6, p. 16",
        "A minimum detectable difference is fixed for the discrimination, agreement, reader-accuracy and deferral endpoints before the test set is evaluated.",
        (
            "triad_reconciliation.core.config:InferenceConfig",
            "triad_reconciliation.metrics.agreement:cohen_kappa",
        ),
        "execution",
    ),
    Claim(
        "C064",
        "Sec. 5.7, p. 17",
        "Coverage, the ratio of viable records collected, is reported beside discrimination.",
        (
            "triad_reconciliation.deployment.silent_mode:silent_mode_metrics",
            "triad_reconciliation.cohort.splits:coverage_report",
        ),
        "execution",
    ),
    Claim(
        "C065",
        "Sec. 5.8, p. 17",
        "The analysis is deterministic under the recorded initial settings.",
        ("triad_reconciliation.core.seeding:set_seed",),
        "execution",
    ),
    Claim(
        "C066",
        "Sec. 5.8, p. 17 (Code availability)",
        "The release carries a schema for perimetry serialisation.",
        (
            "triad_reconciliation.streams.perimetry:serialise_report",
            "triad_reconciliation.streams.perimetry:build_vocabulary",
        ),
        "execution",
    ),
    Claim(
        "C067",
        "Sec. 5.8, p. 17 (Code availability)",
        "The release carries the definition of how the splits were made.",
        ("triad_reconciliation.cohort.splits:split_calibration",),
        "execution",
    ),
    Claim(
        "C068",
        "Sec. 5.8, p. 17 (Code availability)",
        "The release computes the conformal calibration quantiles.",
        ("triad_reconciliation.metrics.conformal:conformal_quantile",),
        "execution",
    ),
    Claim(
        "C069",
        "Table 5, p. 22",
        "Workflow outcomes are reported as sensitivity at matched specificity, treatment-escalation agreement, review-interval agreement, lead time, reclassification and net benefit.",
        ("triad_reconciliation.deployment.workflow:workflow_outcome",),
        "execution",
    ),
    Claim(
        "C070",
        "Table 5, p. 22",
        "Net benefit comes from decision-curve analysis at a threshold probability.",
        ("triad_reconciliation.deployment.workflow:net_benefit",),
        "execution",
    ),
    Claim(
        "C071",
        "Table 5, p. 22",
        "Subgroup areas are reported by disease stage, perimetry reliability, high myopia, age, sex and recorded ancestry group.",
        ("triad_reconciliation.strata.partitions:all_strata",),
        "execution",
    ),
    Claim(
        "C072",
        "Table 6, p. 23",
        "The silent run reports eligible encounters, scored encounters and the two unscored reasons separately.",
        ("triad_reconciliation.deployment.silent_mode:EncounterRecord",),
        "execution",
    ),
    Claim(
        "C073",
        "Table 6, p. 23",
        "Latency is reported as a median with an interquartile range over scored pairs.",
        ("triad_reconciliation.deployment.silent_mode:silent_mode_metrics",),
        "execution",
    ),
    Claim(
        "C074",
        "Table 6, p. 23",
        "An attribution is emitted with every scored decision and its consistency is audited.",
        ("triad_reconciliation.deployment.silent_mode:SilentModeMetrics",),
        "execution",
    ),
    Claim(
        "C075",
        "Table 6, p. 23",
        "The deferred decisions returned to review have an acceptance rate.",
        ("triad_reconciliation.deployment.silent_mode:deployment_log_summary",),
        "execution",
    ),
    Claim(
        "C076",
        "Table 7, p. 24",
        "Each comparator is listed with its family and exactly what it receives.",
        ("triad_reconciliation.comparators.roster:ComparatorSpec",),
        "mapping",
    ),
    Claim(
        "C077",
        "Table 1 row 2, p. 6",
        "The visual field index trend is a clinical rule over serial perimetry.",
        ("triad_reconciliation.comparators.clinical_rules:visual_field_index_trend",),
        "execution",
    ),
    Claim(
        "C078",
        "Table 1 rows 4-5, p. 6",
        "Structure-function regression is reported in a linear and a kernel variant over the optic-disc summary and the perimetry indices.",
        ("triad_reconciliation.comparators.structure_function:kernel_structure_function",),
        "execution",
    ),
    Claim(
        "C079",
        "Table 1 row 11, p. 7",
        "The multimodal language comparator is prompted zero-shot rather than fitted.",
        ("triad_reconciliation.comparators.zero_shot:zero_shot_scores",),
        "execution",
    ),
    Claim(
        "C080",
        "Sec. 5.5, p. 15",
        "The weight is one plus lambda on a pair the model reads as discordant, with lambda fixed on the development group.",
        ("triad_reconciliation.core.config:TrainConfig.reweighting_strength",),
        "execution",
    ),
)


def _resolve(symbol: str) -> tuple[bool, str]:
    if ":" not in symbol:
        return False, f"{symbol} is not written as module:attribute"
    module_name, attribute = symbol.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        return False, f"cannot import {module_name}: {error}"
    target: object = module
    for part in attribute.split("."):
        if not hasattr(target, part):
            return False, f"{module_name} has no attribute {part}"
        target = getattr(target, part)
    return True, "resolved"


def resolve_claims() -> list[dict[str, object]]:
    """Resolve every symbol and report the claim status each one earns."""

    report: list[dict[str, object]] = []
    for claim in CLAIMS:
        missing: list[str] = []
        details: list[str] = []
        for symbol in claim.symbols:
            ok, message = _resolve(symbol)
            if not ok:
                missing.append(symbol)
                details.append(message)
        status = "PASS" if not missing else "FAIL"
        report.append(
            {
                "id": claim.identifier,
                "location": claim.location,
                "statement": claim.statement,
                "symbols": list(claim.symbols),
                "verification": claim.verification,
                "family": claim.family,
                "missing_symbols": missing,
                "code_present": not missing,
                "details": details,
                "status": status,
            }
        )
    return report


def claims_by_verification(kind: VerificationKind) -> tuple[Claim, ...]:
    return tuple(claim for claim in CLAIMS if claim.verification == kind)
