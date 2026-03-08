# CascadeAnalyzer -- computes CEP-Class via Personalized PageRank.
# Domain service -- PageRank computation delegated to infrastructure (networkx).

from __future__ import annotations

import math

from revgate.domain.entities.gene import Gene
from revgate.domain.value_objects.cep_score import CEPScore


class CascadeAnalyzer:
    """Computes Cascade Expansion Phenotype (CEP-Class) via PageRank.

    The infrastructure layer runs Personalized PageRank in the STRING
    network seeded at top dependency genes and returns the resulting
    PageRank vectors as plain dicts.

    This service computes the entropy of each PageRank vector and
    returns the mean entropy as the CEP score.

    High entropy -> influence spreads broadly (CEP-I).
    Low entropy  -> influence stays local (CEP-III).
    """

    def analyze(
        self,
        top_genes: list[Gene],
        pagerank_vectors: list[dict[str, float]],
        cohort_percentile: float,
    ) -> CEPScore:
        """Compute CEP score and return CEPScore value object.

        Args:
            top_genes:          Top dependency genes used as PageRank seeds.
            pagerank_vectors:   One PageRank dict per seed gene.
                                Each dict maps node -> PageRank probability.
            cohort_percentile:  Percentile position in the analysis cohort [0, 100].
                                Computed externally after all cancer types are scored.

        Returns:
            CEPScore with composite value and percentile-based class.
        """
        if not pagerank_vectors:
            return CEPScore(value=0.0, percentile=cohort_percentile)

        entropies = [self._entropy(vec) for vec in pagerank_vectors]
        mean_entropy = sum(entropies) / len(entropies)

        return CEPScore(value=mean_entropy, percentile=cohort_percentile)

    def _entropy(self, pagerank_vector: dict[str, float]) -> float:
        """Shannon entropy of a PageRank probability distribution.

        H = -sum(p * log2(p)) for p > 0

        Higher entropy = more uniform spread = broader cascade.
        """
        entropy = 0.0
        for prob in pagerank_vector.values():
            if prob > 0.0:
                entropy -= prob * math.log2(prob)
        return entropy
