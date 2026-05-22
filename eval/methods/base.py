"""Common interface for pruning methods.

The harness treats every pruning strategy as a swappable component that
gets attached to a loaded LLaVA-Med model. The unmodified baseline is
also a "method" — it just does nothing.

Lifecycle:
    1. __init__(): store hyperparameters
    2. attach(loaded): install hooks; loaded gives access to model + tokenizer
    3. set_question(question): called once per sample, before generate()
    4. (the model generates; hooks fire and use the current question)
    5. detach(loaded): remove hooks, restoring original behavior

The set_question() side channel exists because forward hooks fire inside
model.generate() where we have no direct way to pass auxiliary inputs.

Similarly, current_input_ids is updated by a pre-forward hook on the
LlavaLlamaModel and read by a pre-forward hook on layer 0, so the
layer-0 hook can find visual-token positions in the input sequence.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch


class PruningMethod(ABC):
    """Abstract base class for visual token pruning methods."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.current_question: Optional[str] = None
        # Captured by a hook on the model; read by the layer-0 pruning hook.
        # Reset to None after each prune to avoid stale reads on the
        # generation's subsequent (KV-cache) forward passes.
        self.current_input_ids: Optional[torch.Tensor] = None

    @abstractmethod
    def attach(self, loaded: Any) -> None:
        ...

    @abstractmethod
    def detach(self, loaded: Any) -> None:
        ...

    def set_question(self, question: str) -> None:
        self.current_question = question

    @property
    @abstractmethod
    def name(self) -> str:
        ...