# Gini coefficient of a dependency score distribution.
# Measures concentration: 1.0 = one gene dominates, 0.0 = uniform.
# Immutable value object.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GiniCoefficient:
    """Immutable Gini coefficient [0.0, 1.0].

    High Gini -> concentrated dependency (e.g. BCR-ABL in CML).
    Low Gini  -> distributed dependency (e.g. PDAC).
    """

    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(
                f"GiniCoefficient must be in [0.0, 1.0], got {self.value}"
            )

    @property
    def is_concentrated(self) -> bool:
        """True if Gini > 0.7 -- single dominant dependency."""
        return self.value > 0.7

    @property
    def is_distributed(self) -> bool:
        """True if Gini <= 0.4 -- no dominant dependency."""
        return self.value <= 0.4

    def __repr__(self) -> str:
        return f"GiniCoefficient(value={self.value:.4f})"
