# Exceptions related to tumor classification failures and invariant violations.

from revgate.domain.exceptions.base import RevGateException


class ClassificationFailedError(RevGateException):
    """Raised when a classification axis cannot be assigned."""

    def __init__(self, axis: str, cancer_id: str, reason: str) -> None:
        self.axis = axis
        self.cancer_id = cancer_id
        self.reason = reason
        super().__init__(
            f"Classification failed for axis={axis}, cancer={cancer_id}: {reason}"
        )


class InvariantViolatedError(RevGateException):
    """Raised when a biological sanity check (Tier 0) fails.
    This is a hard stop -- pipeline must not proceed.
    """

    def __init__(self, invariant_id: str, expected: str, actual: str) -> None:
        self.invariant_id = invariant_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Biological invariant {invariant_id} violated: "
            f"expected={expected}, actual={actual}"
        )


class InsufficientDataError(RevGateException):
    """Raised when a cancer type has too few cell lines or patients."""

    def __init__(self, cancer_id: str, required: int, available: int) -> None:
        self.cancer_id = cancer_id
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient data for {cancer_id}: required={required}, available={available}"
        )
