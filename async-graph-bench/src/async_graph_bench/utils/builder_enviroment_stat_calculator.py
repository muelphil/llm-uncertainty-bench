from abc import ABC

class BuilderEnvironment(ABC):
    """Environment seen by all stat calculators when they are built in the UEManager. Stat calculators can share the constructed objects via the environment."""

    def __init__(self):
        pass
