"""SLAKE dataset loader.

SLAKE (Liu et al., 2021): ~14,000 question-answer pairs over 642 medical
images (bilingual: English + Chinese). This project uses the English-only
subset, filtered by q_lang == 'en'.

This loader targets the original SLAKE distribution (BoKelvin/SLAKE on
HuggingFace), which is the raw OSF distribution layout:

    root/
    ├── train.json
    ├── validation.json
    ├── test.json
    └── imgs/
        ├── xmlab0/source.jpg
        ├── xmlab1/source.jpg
        └── ...

Each JSON file is a list of records with these fields:
    img_id, img_name, question, answer, q_lang, location,
    modality, answer_type, base_type, content_type, triple, qid

Two adaptations relative to the raw format:

1. The bilingual data is filtered to q_lang == 'en' only.
2. answer_type is normalised: SLAKE uses uppercase ("OPEN", "CLOSED")
   while the VQASample contract uses lowercase ("open", "closed").

Images are already loose files on disk (after the one-time imgs.zip
extraction done during data setup), so no bytes-to-disk materialisation
is needed -- the loader resolves <root>/imgs/<img_name> directly.
"""

import json
from pathlib import Path
from typing import Iterator, List

from .base import MedVQADataset, VQASample


# Map our contract's split names to the JSON filenames SLAKE actually ships.
# SLAKE uses "validation" instead of "val"; we accept both for convenience.
_SPLIT_FILE = {
    "train": "train.json",
    "val": "validation.json",
    "validation": "validation.json",
    "test": "test.json",
}


class SlakeDataset(MedVQADataset):
    """Loader for SLAKE (English subset)."""

    @property
    def name(self) -> str:
        return "slake"

    def _json_path(self) -> Path:
        """Resolve the split argument to the right JSON file."""
        if self.split not in _SPLIT_FILE:
            raise ValueError(
                f"Unknown split {self.split!r}. Expected one of "
                f"{sorted(_SPLIT_FILE)}."
            )
        path = self.root / _SPLIT_FILE[self.split]
        if not path.exists():
            raise FileNotFoundError(
                f"SLAKE split file not found: {path}. "
                f"Check that --dataset-root points at the SLAKE directory "
                f"(it should contain train.json, validation.json, test.json, "
                f"and an imgs/ subfolder)."
            )
        return path

    def _images_dir(self) -> Path:
        """Verify the imgs/ directory exists and return its path."""
        path = self.root / "imgs"
        if not path.is_dir():
            raise FileNotFoundError(
                f"SLAKE images directory not found: {path}. "
                f"Did you unzip imgs.zip? Run: "
                f"cd {self.root} && unzip imgs.zip"
            )
        return path

    def _load_samples(self) -> List[VQASample]:
        """Read the JSON, filter to English, build VQASample objects.

        Done eagerly (all at once) -- SLAKE test is only 1,061 English rows,
        which fits in memory trivially, and an eager list makes __len__
        exact and iteration restartable.
        """
        json_path = self._json_path()
        images_dir = self._images_dir()

        with open(json_path) as f:
            all_records = json.load(f)

        # Filter to English-only. The Chinese half is structurally identical
        # but linguistically out of scope for this project.
        english = [r for r in all_records if r.get("q_lang") == "en"]
        if not english:
            raise RuntimeError(
                f"Found 0 English records in {json_path}. "
                f"Either the q_lang field is missing, or the split is empty."
            )

        # Sanity-check the first image exists. Failing now (before iteration)
        # gives a clear error message instead of a mysterious failure
        # 200 samples into evaluation.
        first_img = images_dir / english[0]["img_name"]
        if not first_img.exists():
            raise FileNotFoundError(
                f"First image path does not exist: {first_img}. "
                f"This suggests the imgs/ directory layout differs from the "
                f"expected <root>/imgs/<img_name> structure."
            )

        samples: List[VQASample] = []
        for rec in english:
            samples.append(
                VQASample(
                    # qid is unique within the dataset. Prefix with dataset
                    # and split so question_ids are globally meaningful in
                    # merged prediction files later on.
                    question_id=f"slake_{self.split}_{rec['qid']}",
                    image_path=str(images_dir / rec["img_name"]),
                    question=rec["question"],
                    # Some answers are numbers (e.g. "how many lobes...?" -> 3).
                    # Coerce to string for a consistent contract.
                    answer=str(rec["answer"]),
                    # Normalise OPEN/CLOSED -> open/closed to match the
                    # VQASample contract.
                    answer_type=rec["answer_type"].lower(),
                    dataset="slake",
                    metadata={
                        "location": rec.get("location"),
                        "modality": rec.get("modality"),
                        "content_type": rec.get("content_type"),
                        "base_type": rec.get("base_type"),
                        "img_id": rec.get("img_id"),
                    },
                )
            )

        if self.max_samples is not None:
            samples = samples[: self.max_samples]

        return samples

    def __iter__(self) -> Iterator[VQASample]:
        # _load_samples is cheap enough to call here; matches the VQA-RAD
        # loader's pattern. If iteration happened many times we'd cache it,
        # but the runner iterates exactly once.
        return iter(self._load_samples())

    def __len__(self) -> int:
        return len(self._load_samples())