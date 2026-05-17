"""Common interface for medical VQA datasets.

Every benchmark (VQA-RAD, SLAKE, PathVQA) implements this interface so the
runner can iterate over them uniformly without knowing benchmark-specific
details.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class VQASample:
    """One question-answer pair tied to one image.

    Attributes:
        question_id: Unique identifier within the dataset. Used to key
            per-question predictions in the output JSONL.
        image_path: Absolute path to the image file on disk.
        question: The natural language question.
        answer: The ground-truth answer (a string; for closed-ended
            yes/no questions this is "yes" or "no").
        answer_type: One of "closed" (yes/no, multiple choice) or "open"
            (free-form). Drives which accuracy metric applies.
        dataset: Name of the source dataset ("vqa_rad", "slake",
            "path_vqa"). Useful when merging predictions across datasets.
        metadata: Any extra fields (modality, organ, difficulty, etc.)
            that a specific dataset wants to expose for analysis.
    """
    question_id: str
    image_path: str
    question: str
    answer: str
    answer_type: str  # "closed" or "open"
    dataset: str
    metadata: Optional[dict] = None


class MedVQADataset(ABC):
    """Abstract base class for medical VQA benchmarks.

    Subclasses load their specific format and expose a uniform iterator
    of VQASample objects. The runner doesn't care which benchmark it's
    iterating over.
    """

    def __init__(self, root: str, split: str = "test", max_samples: Optional[int] = None):
        """
        Args:
            root: Path to the dataset's root directory on disk.
            split: Which split to load ("train", "val", "test"). Most
                evaluation uses "test"; "val" is useful for development
                so you don't tune on the actual test set.
            max_samples: If set, truncate to this many samples. Useful
                for fast smoke tests during harness development.
        """
        self.root = Path(root)
        self.split = split
        self.max_samples = max_samples

    @abstractmethod
    def __iter__(self) -> Iterator[VQASample]:
        """Yield VQASample objects one at a time."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples in this split (after max_samples cap)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset identifier used in output files."""
        ...
