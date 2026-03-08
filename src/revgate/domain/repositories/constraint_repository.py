# Port interface for gnomAD evolutionary constraint data access.
# Infrastructure layer implements this protocol.

from typing import Protocol, runtime_checkable


@runtime_checkable
class ConstraintRepository(Protocol):
    """Abstract interface for gnomAD v4 gene constraint metrics.

    The concrete implementation downloads and caches the gnomAD v4
    constraint TSV file from gnomad.broadinstitute.org.

    Methods:
        get_pli_scores   -- pLI scores for all genes
        get_loeuf_scores -- LOEUF scores for all genes
    """

    def get_pli_scores(self) -> dict[str, float]:
        """Return pLI scores for all genes in gnomAD v4.

        pLI = probability of loss-of-function intolerance.
        High pLI (>0.9) means the gene is intolerant to LoF variants --
        the tumor cannot easily mutate around this dependency.

        Returns:
            Dict mapping gene symbol -> pLI score [0.0, 1.0].
            Missing genes are not included in the dict.
        """
        ...

    def get_loeuf_scores(self) -> dict[str, float]:
        """Return LOEUF scores for all genes in gnomAD v4.

        LOEUF = loss-of-function observed/expected upper bound.
        Low LOEUF (<0.35) indicates strong constraint.

        Returns:
            Dict mapping gene symbol -> LOEUF score.
            Missing genes are not included in the dict.
        """
        ...
