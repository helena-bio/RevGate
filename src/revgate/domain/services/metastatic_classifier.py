# MetastaticClassifier -- assigns MP-Class from RDP-Class and EMT markers.
# Pure domain service.

from __future__ import annotations

from revgate.domain.value_objects.mp_score import MPScore, EMT_MARKERS
from revgate.domain.value_objects.rdp_score import RDPClass


class MetastaticClassifier:
    """Assigns MP-Class (MP-I / MP-II / MP-III) from expression data.

    Combines:
        1. RDP-III (EMT) enrichment score from ssGSEA
        2. Mean z-score of EMT invasion markers:
           TWIST1, SNAI1, ZEB1, VIM, CDH2, MMP2, MMP9

    The biological logic:
        EMT program + high invasion markers -> MP-I (high metastatic risk)
        Lineage program + low invasion      -> MP-III (low risk)
        All other combinations              -> MP-II (moderate)
    """

    def classify(
        self,
        rdp_emt_enrichment: float,
        expression_zscores: dict[str, float],
    ) -> MPScore:
        """Compute MP score and return MPScore value object.

        Args:
            rdp_emt_enrichment:  NES of RDP-III (EMT) program from ssGSEA.
            expression_zscores:  Dict of gene symbol -> z-score from TCGA expression.
                                 Only EMT_MARKERS genes are used.

        Returns:
            MPScore value object with mp_class assigned.
        """
        invasion_zscore = self._compute_invasion_zscore(expression_zscores)

        return MPScore(
            rdp_emt_enrichment=rdp_emt_enrichment,
            invasion_zscore=invasion_zscore,
        )

    def _compute_invasion_zscore(
        self,
        expression_zscores: dict[str, float],
    ) -> float:
        """Compute mean z-score across EMT invasion marker genes.

        Genes missing from expression data are excluded from the mean.
        If no markers are available, returns 0.0 (neutral).
        """
        scores = [
            expression_zscores[marker]
            for marker in EMT_MARKERS
            if marker in expression_zscores
        ]

        if not scores:
            return 0.0

        return sum(scores) / len(scores)
