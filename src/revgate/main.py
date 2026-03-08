# RevGate service entry point.
# Imported by the root main.py CLI.

from revgate.config.settings import RevGateSettings
from revgate.di.dependencies import Container


def create_container(settings: RevGateSettings | None = None) -> Container:
    """Build and return the fully wired DI container.

    Args:
        settings: Optional pre-built settings. Loads from YAML if None.

    Returns:
        Container with all repositories and use cases wired.
    """
    return Container.build(settings=settings)
