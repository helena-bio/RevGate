# DependencyProfiler -- constructs dependency profile for a cancer type.
# Pure domain service -- no external I/O, no pandas dependency.
# Operates only on domain entities and value objects.

from __future__ import annotations

from dataclasses import dataclass, field

from revgate.domain.entities.gene import Gene
from revgate.domain.value_objects.dependency_score import DependencyScore, ESSENTIAL_THRESHOLD
from revgate.domain.value_objects.gini_coefficient import GiniCoefficient


# Default number of top dependency genes to consider
DEFAULT_TOP_N = 20


@dataclass
class DependencyProfile:
    """Aggregated dependency landscape for a single cancer type.

    Attributes:
        cancer_id       -- TCGA cancer type code
        top_genes       -- top-N genes ranked by mean dependency strength
        mean_scores     -- mean Chronos score per gene symbol
        gini            -- Gini coefficient of top-N dependency distribution
    """

    cancer_id: str
    top_genes: list[Gene] = field(default_factory=list)
    mean_scores: dict[str, float] = field(default_factory=dict)
    gini: GiniCoefficient = field(default_factory=lambda: GiniCoefficient(0.0))

    @property
    def top_n(self) -> int:
        return len(self.top_genes)

    @property
    def has_dominant_dependency(self) -> bool:
        """True if top gene Chronos score < -1.5 (strongly selective)."""
        if not self.top_genes:
            return False
        top_symbol = self.top_genes[0].symbol
        score = self.mean_scores.get(top_symbol, 0.0)
        return score < -1.5


class DependencyProfiler:
    """Constructs the dependency profile for a cancer type from DepMap data.

    Operates on pre-loaded Gene entities with dependency_scores populated.
    No I/O -- data loading is handled by the infrastructure layer.
    """

    def profile(
        self,
        cancer_id: str,
        genes: list[Gene],
        top_n: int = DEFAULT_TOP_N,
    ) -> DependencyProfile:
        """Build dependency profile for a cancer type.

        Args:
            cancer_id: TCGA cancer type code.
            genes:     List of Gene entities with dependency_scores for cancer_id.
            top_n:     Number of top dependency genes to retain.

        Returns:
            DependencyProfile with ranked top genes and Gini coefficient.
        """
        # Collect mean Chronos scores for this cancer type
        mean_scores: dict[str, float] = {}
        for gene in genes:
            score = gene.get_dependency(cancer_id)
            if score is not None:
                mean_scores[gene.symbol] = score.value

        if not mean_scores:
            return DependencyProfile(cancer_id=cancer_id)

        # Rank by dependency strength (most negative = strongest dependency)
        ranked_symbols = sorted(mean_scores, key=lambda s: mean_scores[s])

        # Take top-N
        top_symbols = set(ranked_symbols[:top_n])
        top_genes = [g for g in genes if g.symbol in top_symbols]
        top_genes.sort(key=lambda g: mean_scores.get(g.symbol, 0.0))

        # Compute Gini on ALL gene scores -- concentration across full landscape
        # Top-N is used for pLI/centrality components, not for Gini
        all_scores = [mean_scores[s] for s in ranked_symbols]
        gini = self.compute_gini(all_scores)

        return DependencyProfile(
            cancer_id=cancer_id,
            top_genes=top_genes,
            mean_scores=mean_scores,
            gini=gini,
        )

    def compute_gini(self, scores: list[float]) -> GiniCoefficient:
        """Compute Gini coefficient of a dependency score distribution.

        Uses absolute values of Chronos scores so that more negative
        scores contribute more to concentration.

        Args:
            scores: List of Chronos scores (typically negative floats).

        Returns:
            GiniCoefficient value object in [0.0, 1.0].
        """
        if not scores:
            return GiniCoefficient(0.0)

        # Work with absolute values -- stronger dependency = larger magnitude
        values = sorted(abs(s) for s in scores)
        n = len(values)

        if n == 1:
            return GiniCoefficient(1.0)

        total = sum(values)
        if total == 0.0:
            return GiniCoefficient(0.0)

        # Standard Gini formula: G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n+1)/n
        weighted_sum = sum((i + 1) * v for i, v in enumerate(values))
        gini_value = (2.0 * weighted_sum) / (n * total) - (n + 1) / n

        # Clamp to [0.0, 1.0] to handle floating point edge cases
        gini_value = max(0.0, min(1.0, gini_value))

        return GiniCoefficient(gini_value)
