# Base repository protocol.
# All domain repository interfaces inherit from this marker protocol.
# No concrete implementations here -- infrastructure layer implements these.

from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseRepository(Protocol):
    """Marker protocol for all RevGate repositories.

    Domain layer depends only on these Protocol interfaces.
    Infrastructure layer provides concrete implementations.
    Dependency rule: infrastructure -> domain, never the reverse.
    """
    ...
