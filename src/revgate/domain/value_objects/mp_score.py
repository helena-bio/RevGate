# Metastatic Propensity score -- integrates RDP-Class with EMT invasion markers.
# Immutable value object.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MPClass(str, Enum):
    """Metastatic Propensity classification."""

    MP_I   = "MP-I"    # High risk > 70% 5-year metastasis
    MP_II  = "MP-II"   # Moderate risk 30-70%
    MP_III = "MP-III"  # Low risk < 30%


# EMT invasion marker genes used for mean z-score computation
EMT_MARKERS = ("TWIST1", "SNAI1", "ZEB1", "VIM", "CDH2", "MMP2", "MMP9")

# Thresholds for MP classification
MP_I_THRESHOLD  = 1.0   # mean z-score above this -> MP-I
MP_III_THRESHOLD = -0.5  # mean z-score below this -> MP-III


@dataclass(frozen=True)
class MPScore:
    """Metastatic Propensity Score.

    Combines:
        rdp_emt_enrichment -- NES of EMT program (RDP-III) from ssGSEA
        invasion_zscore    -- mean z-score of EMT invasion markers
    """

    rdp_emt_enrichment: float
    invasion_zscore: float

    @property
    def mp_class(self) -> MPClass:
        """Classify based on invasion z-score combined with EMT enrichment.

        Logic from HLD:
            RDP-III (EMT) + high invasion -> MP-I
            Non-EMT + moderate invasion   -> MP-II
            Lineage program + low invasion -> MP-III
        """
        if self.invasion_zscore > MP_I_THRESHOLD and self.rdp_emt_enrichment >= 2.0:
            return MPClass.MP_I
        if self.invasion_zscore < MP_III_THRESHOLD and self.rdp_emt_enrichment < 1.0:
            return MPClass.MP_III
        return MPClass.MP_II

    def __repr__(self) -> str:
        return (
            f"MPScore(invasion_zscore={self.invasion_zscore:.4f}, "
            f"rdp_emt={self.rdp_emt_enrichment:.4f}, "
            f"class={self.mp_class.value})"
        )
