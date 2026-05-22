"""Baseline 'method' that does nothing.

Running the harness with this method produces the unmodified-model
baseline numbers that every other method is compared against.
"""

from .base import PruningMethod


class BaselineMethod(PruningMethod):
    """No-op method: model runs unchanged."""

    @property
    def name(self) -> str:
        return "baseline"

    def attach(self, loaded) -> None:
        pass

    def detach(self, loaded) -> None:
        pass