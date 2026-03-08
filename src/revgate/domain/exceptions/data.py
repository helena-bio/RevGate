# Exceptions related to data availability and download failures.

from revgate.domain.exceptions.base import RevGateException


class DataNotAvailableError(RevGateException):
    """Raised when a required data source is not cached or downloadable."""

    def __init__(self, source: str) -> None:
        self.source = source
        super().__init__(f"Data source not available: {source}")


class DownloadFailedError(RevGateException):
    """Raised when a data download fails after retries."""

    def __init__(self, source: str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"Download failed for {source}: {reason}")


class CacheCorruptedError(RevGateException):
    """Raised when a cached file fails checksum validation."""

    def __init__(self, path: str, expected: str, actual: str) -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Cache corrupted at {path}: expected SHA256={expected}, got={actual}"
        )
