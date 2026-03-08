# DevelopmentalClassifier -- assigns RDP-Class via ssGSEA enrichment.
# Domain service -- orchestrates enrichment logic.
# Actual ssGSEA computation delegated to infrastructure (gseapy).

from __future__ import annotations

from revgate.domain.value_objects.rdp_score import RDPClass, RDPScore, NES_THRESHOLD, FDR_THRESHOLD


class DevelopmentalClassifier:
    """Assigns RDP-Class (RDP-I through RDP-V) from ssGSEA enrichment results.

    The ssGSEA computation itself is performed by the infrastructure layer
    (gseapy). This service receives the enrichment results as plain dicts
    and applies the classification logic.

    RDP assignment rule:
        1. Collect all programs with NES >= 2.0 AND FDR < 0.05
        2. If any significant, assign the one with highest NES
        3. If none significant, assign the one with highest NES regardless
    """

    def classify(
        self,
        enrichment_results: dict[str, dict[str, float]],
    ) -> list[RDPScore]:
        """Classify developmental program from ssGSEA enrichment results.

        Args:
            enrichment_results: Dict mapping RDPClass value string ->
                                 {'nes': float, 'fdr': float}
                                 e.g. {'RDP-I': {'nes': 2.3, 'fdr': 0.01}}

        Returns:
            List of RDPScore value objects, one per program.
            Sorted by NES descending.
        """
        rdp_scores: list[RDPScore] = []

        for program_str, metrics in enrichment_results.items():
            try:
                program = RDPClass(program_str)
            except ValueError:
                # Unknown program key -- skip silently
                continue

            nes = float(metrics.get("nes", 0.0))
            fdr = float(metrics.get("fdr", 1.0))

            rdp_scores.append(RDPScore(program=program, nes=nes, fdr=fdr))

        # Sort by NES descending -- highest enrichment first
        rdp_scores.sort(key=lambda s: s.nes, reverse=True)

        return rdp_scores

    def get_primary_class(self, rdp_scores: list[RDPScore]) -> RDPClass:
        """Return the primary RDP-Class from a list of scored programs.

        Args:
            rdp_scores: List of RDPScore value objects.

        Returns:
            RDPClass of the highest significant program,
            or highest NES program if none are significant.
        """
        if not rdp_scores:
            raise ValueError("Cannot assign RDP-Class from empty scores list")

        significant = [s for s in rdp_scores if s.is_significant]
        pool = significant if significant else rdp_scores

        return max(pool, key=lambda s: s.nes).program
