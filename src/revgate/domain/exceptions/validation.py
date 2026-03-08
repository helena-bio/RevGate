# Exceptions related to input validation and sanity check failures.

from revgate.domain.exceptions.base import RevGateException


class ValidationError(RevGateException):
    """Raised when input data fails structural validation."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"Validation error on field={field}: {reason}")


class SanityCheckFailedError(RevGateException):
    """Raised when a pipeline sanity check does not pass.
    Distinct from InvariantViolatedError -- sanity checks are
    configurable warnings, invariants are hard stops.
    """

    def __init__(self, check: str, details: str) -> None:
        self.check = check
        self.details = details
        super().__init__(f"Sanity check failed [{check}]: {details}")
