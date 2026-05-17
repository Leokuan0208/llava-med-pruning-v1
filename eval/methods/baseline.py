"""Baseline 'method' that does nothing.

This is the first method to implement because it's the simplest possible
attach/detach pair: both are no-ops. Running the harness with this method
produces the unmodified-model baseline numbers that every other method is
compared against.
"""

from .base import PruningMethod


class BaselineMethod(PruningMethod):
    """No-op method: model runs unchanged."""

    @property
    def name(self) -> str:
        return "baseline"

    def attach(self, model) -> None:
        # Nothing to do; the model is already in its unmodified state.
        pass

    def detach(self, model) -> None:
        # Nothing was attached, so nothing to detach.
        pass
