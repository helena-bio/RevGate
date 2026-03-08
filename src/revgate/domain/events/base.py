# Base domain event.
# All RevGate pipeline events inherit from DomainEvent.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all RevGate domain events.

    Immutable -- events are facts that happened, never mutated.

    Attributes:
        occurred_at -- UTC timestamp when the event was created
    """

    occurred_at: datetime = field(default_factory=datetime.utcnow)

    def event_name(self) -> str:
        """Return the event class name as a string identifier."""
        return self.__class__.__name__
