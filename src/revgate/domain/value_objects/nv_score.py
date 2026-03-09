# Network Vulnerability Score -- composite of Gini, selectivity, constraint, centrality.
# Formula: NV-Score = 0.25*Gini + 0.25*selectivity + 0.25*mean_pLI + 0.25*mean_centrality
#
# Weight rationale (Finding_04, 2026-03-09):
#   PCA on 20 DepMap 24Q4 lineages yielded PC1 loadings [0.26, 0.20, 0.36, 0.18],
#   but permutation test showed p=0.787 -- PCA structure is not statistically significant.
#   Bootstrap resampling (n=1000) confirmed instability (CV 0.40-0.50 per weight).
#   Equal weights adopted: Cohen's kappa (PCA vs Equal) = 0.780, 19/20 lineages identical.
#   PCA weights retained in Supplementary as exploratory sensitivity analysis.
# Immutable value object.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NVClass(str, Enum):
    """Network Vulnerability classification."""

    NV_A = "NV-A"  # Single-node vulnerability -- targeted monotherapy candidate
    NV_B = "NV-B"  # Moderate concentration -- combination targeted therapy
    NV_C = "NV-C"  # Distributed -- chemotherapy / immunotherapy


# Classification thresholds -- recalibrated on 20 tumor lineages (DepMap 24Q4)
# with equal weights (0.25 per component). Finding_04, 2026-03-09.
# Biological anchors (equal-weight scores, MinMax-scaled components):
#   NV-A (>= 0.50): Myeloid (MYB/CBFB, 0.821), Kidney (HNF1B, 0.794),
#                   Skin (SOX10/BRAF, 0.789), Breast (FOXA1, 0.711)
#   NV-B (0.35-0.50): Soft Tissue (0.437), CNS/Brain (0.411),
#                     Peripheral Nervous System (0.469)
#   NV-C (< 0.35):  Testis (0.223) -- distributed housekeeping dependencies
NV_A_THRESHOLD = 0.50
NV_C_THRESHOLD = 0.35


@dataclass(frozen=True)
class NVScore:
    """Composite Network Vulnerability Score.

    Four components (all in [0, 1]):
        gini             -- dependency concentration across all genes (from DepMap)
        mean_selectivity -- cancer-type specificity of top-N genes vs other cancers
        mean_constraint  -- mean pLI of top-N dependent genes (from gnomAD)
        mean_centrality  -- mean degree centrality of top-N genes (from STRING)

    Composite: equal-weighted sum (0.25 per component).
    Rationale: permutation test on 20-lineage PCA showed p=0.787 (not significant);
    bootstrap CV 0.40-0.50 for all PCA weights. Equal weights are methodologically
    justified given near-orthogonal component structure (max Pearson r=0.32).
    Cohen's kappa (PCA vs Equal) = 0.780; 19/20 lineages classify identically.
    See: Finding_04_NVScore_Weight_Stability_Analysis.md

    All four components are independent axes of tumor vulnerability:
    Gini (concentration), Selectivity (specificity), pLI (constraint), Centrality (network).
    """

    gini: float
    mean_selectivity: float
    mean_constraint: float
    mean_centrality: float

    def __post_init__(self) -> None:
        for name, val in [
            ("gini", self.gini),
            ("mean_selectivity", self.mean_selectivity),
            ("mean_constraint", self.mean_constraint),
            ("mean_centrality", self.mean_centrality),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"NVScore.{name} must be in [0.0, 1.0], got {val}")

    @property
    def composite(self) -> float:
        """NV-Score = 0.25*Gini + 0.25*selectivity + 0.25*pLI + 0.25*centrality.

        Equal weights adopted after permutation test showed PCA structure is not
        statistically significant (p=0.787, n=1000 permutations, n=20 lineages).
        Bootstrap CI for PCA weights: CV 0.40-0.50 (unstable). Finding_04.
        """
        return (
            0.25 * self.gini
            + 0.25 * self.mean_selectivity
            + 0.25 * self.mean_constraint
            + 0.25 * self.mean_centrality
        )

    @property
    def nv_class(self) -> NVClass:
        """Classify into NV-A / NV-B / NV-C based on composite score."""
        score = self.composite
        if score >= NV_A_THRESHOLD:
            return NVClass.NV_A
        if score < NV_C_THRESHOLD:
            return NVClass.NV_C
        return NVClass.NV_B

    def __repr__(self) -> str:
        return (
            f"NVScore(composite={self.composite:.4f}, class={self.nv_class.value}, "
            f"gini={self.gini:.4f}, selectivity={self.mean_selectivity:.4f}, "
            f"constraint={self.mean_constraint:.4f}, centrality={self.mean_centrality:.4f})"
        )
