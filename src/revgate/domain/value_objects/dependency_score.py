# Wraps a single DepMap Chronos dependency score.
# Negative values indicate dependency (cell dies without the gene).
# Immutable value object -- no identity, equality by value.

from dataclasses import dataclass


# DepMap standard threshold for essentiality
ESSENTIAL_THRESHOLD = -0.5

# Strong selective dependency threshold
SELECTIVE_THRESHOLD = -1.0


@dataclass(frozen=True)
class DependencyScore:
    """Immutable wrapper around a DepMap Chronos score.

    Chronos score semantics:
        0.0  -> no effect on cell viability
       -0.5  -> essential (cell significantly impaired)
       -1.0  -> strongly selective dependency
    """

    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"DependencyScore.value must be numeric, got {type(self.value)}")

    @property
    def is_essential(self) -> bool:
        """True if Chronos score < -0.5 (DepMap standard threshold)."""
        return self.value < ESSENTIAL_THRESHOLD

    @property
    def is_strongly_selective(self) -> bool:
        """True if Chronos score < -1.0 (strong selective dependency)."""
        return self.value < SELECTIVE_THRESHOLD

    def __repr__(self) -> str:
        return (
            f"DependencyScore(value={self.value:.4f}, "
            f"essential={self.is_essential}, "
            f"selective={self.is_strongly_selective})"
        )
