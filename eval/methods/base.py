"""Common interface for pruning methods.

The harness treats every pruning strategy as a swappable component that
gets attached to a loaded LLaVA-Med model. The unmodified baseline is
also a "method" — it just does nothing.

The interface is designed around the FastV-style hook pattern: pruning
happens inside the LLM's forward pass at a configurable layer. A method
attaches hooks when applied and removes them when detached, so the same
model object can be evaluated under multiple methods sequentially.
"""

from abc import ABC, abstractmethod
from typing import Any


class PruningMethod(ABC):
    """Abstract base class for visual token pruning methods.

    Lifecycle:
        1. __init__(): store hyperparameters
        2. attach(model): install hooks on the given LLaVA-Med model
        3. (evaluation runs)
        4. detach(model): remove hooks, restoring the original behavior

    The attach/detach split lets the runner evaluate multiple methods
    on the same loaded model without reloading weights between runs.
    """

    def __init__(self, **kwargs):
        """Store hyperparameters. Subclasses override to validate args."""
        self.config = kwargs

    @abstractmethod
    def attach(self, model: Any) -> None:
        """Install hooks/modifications on the model.

        Called once before evaluation begins. The model is the loaded
        LLaVA-Med model from llava.model.builder.load_pretrained_model.
        """
        ...

    @abstractmethod
    def detach(self, model: Any) -> None:
        """Remove hooks/modifications, restoring original behavior.

        Called once after evaluation ends. Must leave the model in a
        state where it would produce identical outputs to the
        unmodified model.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Method identifier used in output files (e.g., 'fastv', 'ours')."""
        ...
