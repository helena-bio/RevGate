# Gene entity -- represents a gene with its dependency and network properties.
# No external imports -- pure domain object.

from __future__ import annotations

from dataclasses import dataclass, field

from revgate.domain.value_objects.dependency_score import DependencyScore


@dataclass
class Gene:
    """A gene with its associated dependency and network properties.

    Attributes:
        gene_id          -- Ensembl gene ID (e.g. ENSG00000157764)
        symbol           -- HUGO symbol (e.g. BRAF)
        pli_score        -- gnomAD probability of LoF intolerance [0.0, 1.0]
        loeuf_score      -- gnomAD LoF observed/expected upper bound
        network_degree   -- STRING PPI degree centrality (raw count)
        dependency_scores -- Chronos scores per cancer_id string key
    """

    gene_id: str
    symbol: str
    pli_score: float = 0.0
    loeuf_score: float = 1.0
    network_degree: int = 0
    dependency_scores: dict[str, DependencyScore] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.gene_id:
            raise ValueError("Gene.gene_id must not be empty")
        if not self.symbol:
            raise ValueError("Gene.symbol must not be empty")
        if not (0.0 <= self.pli_score <= 1.0):
            raise ValueError(f"Gene.pli_score must be in [0.0, 1.0], got {self.pli_score}")
        if self.network_degree < 0:
            raise ValueError(f"Gene.network_degree must be >= 0, got {self.network_degree}")

    def get_dependency(self, cancer_id: str) -> DependencyScore | None:
        """Return dependency score for a given cancer type, or None."""
        return self.dependency_scores.get(cancer_id)

    def is_essential_in(self, cancer_id: str) -> bool:
        """True if gene is essential (Chronos < -0.5) in the given cancer type."""
        score = self.get_dependency(cancer_id)
        return score.is_essential if score is not None else False

    def __repr__(self) -> str:
        return (
            f"Gene(symbol={self.symbol!r}, gene_id={self.gene_id!r}, "
            f"pli={self.pli_score:.3f}, degree={self.network_degree})"
        )

    def __hash__(self) -> int:
        return hash(self.gene_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gene):
            return NotImplemented
        return self.gene_id == other.gene_id
