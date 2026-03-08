# Network Vulnerability Score -- composite of Gini, selectivity, constraint, centrality.
# Formula: NV-Score = 0.26*Gini + 0.20*selectivity + 0.36*mean_pLI + 0.18*mean_centrality
# Weights derived from PCA on 20 DepMap 24Q4 lineages (PC1 loadings, normalized absolute values).
# Immutable value object.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NVClass(str, Enum):
    """Network Vulnerability classification."""

    NV_A = "NV-A"  # Single-node vulnerability -- targeted monotherapy candidate
    NV_B = "NV-B"  # Moderate concentration -- combination targeted therapy
    NV_C = "NV-C"  # Distributed -- chemotherapy / immunotherapy


# Classification thresholds -- recalibrated on 25 tumor lineages (DepMap 24Q4)
# after PCA weight derivation. Biological anchors:
#   NV-A: Skin (BRAF, 0.588), Kidney (HNF1B, 0.558), Myeloid (MYB/CBFB, 0.538)
#   NV-B: Pancreas (KRAS, 0.480), Lymphoid (IRF4, 0.481)
#   NV-C: Lung (0.387), CNS/Brain (0.361)
NV_A_THRESHOLD = 0.53
NV_C_THRESHOLD = 0.41


@dataclass(frozen=True)
class NVScore:
    """Composite Network Vulnerability Score.

    Four components (all in [0, 1]):
        gini             -- dependency concentration across all genes (from DepMap)
        mean_selectivity -- cancer-type specificity of top-N genes vs other cancers
        mean_constraint  -- mean pLI of top-N dependent genes (from gnomAD)
        mean_centrality  -- mean degree centrality of top-N genes (from STRING)

    Composite: weighted sum derived from PCA on 20 tumor lineages (DepMap 24Q4).
    PC1 loadings (normalized absolute values):
        gini: 0.260, selectivity: 0.201, mean_pLI: 0.356, mean_centrality: 0.183

    pLI (constraint) is the primary discriminator -- essential gene dependency
    reflects the DNV-TC hypothesis that tumors retain ontogenetic vulnerabilities.
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
        """NV-Score = 0.26*Gini + 0.20*selectivity + 0.36*pLI + 0.18*centrality.

        Weights from PCA PC1 loadings on 20 DepMap 24Q4 tumor lineages.
        """
        return (
            0.26 * self.gini
            + 0.20 * self.mean_selectivity
            + 0.36 * self.mean_constraint
            + 0.18 * self.mean_centrality
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
