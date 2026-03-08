# Base exception for all RevGate domain errors.
# All service-specific exceptions inherit from RevGateException.


class RevGateException(Exception):
    """Base class for all RevGate exceptions."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r})"
