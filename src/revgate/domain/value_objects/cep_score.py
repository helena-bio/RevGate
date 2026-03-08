# Cascade Expansion Phenotype score -- PageRank-based network propagation.
# Immutable value object.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CEPClass(str, Enum):
    """Cascade Expansion Phenotype classification."""

    CEP_I   = "CEP-I"    # Broad cascade -- rapid network-wide effects
    CEP_II  = "CEP-II"   # Moderate cascade
    CEP_III = "CEP-III"  # Contained -- limited propagation


@dataclass(frozen=True)
class CEPScore:
    """Personalized PageRank cascade spread score.

    The score is the mean entropy of PageRank vectors seeded at
    the top-20 dependency genes in the STRING PPI network.

    High score -> perturbation cascades broadly (CEP-I).
    Low score  -> perturbation stays local (CEP-III).

    Percentile thresholds are computed at runtime from the
    distribution across all cancer types in the analysis cohort.
    """

    value: float
    percentile: float  # Position in the cohort distribution [0.0, 100.0]

    def __post_init__(self) -> None:
        if self.value < 0.0:
            raise ValueError(f"CEPScore.value must be non-negative, got {self.value}")
        if not (0.0 <= self.percentile <= 100.0):
            raise ValueError(
                f"CEPScore.percentile must be in [0.0, 100.0], got {self.percentile}"
            )

    @property
    def cep_class(self) -> CEPClass:
        """Classify by percentile position in cohort distribution."""
        if self.percentile >= 75.0:
            return CEPClass.CEP_I
        if self.percentile >= 25.0:
            return CEPClass.CEP_II
        return CEPClass.CEP_III

    def __repr__(self) -> str:
        return (
            f"CEPScore(value={self.value:.4f}, "
            f"percentile={self.percentile:.1f}, "
            f"class={self.cep_class.value})"
        )
