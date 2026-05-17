"""VQA-RAD dataset loader.

VQA-RAD (Lau et al., 2018): ~3,500 question-answer pairs over 315
radiology images. Mix of yes/no (closed) and free-form (open) questions.

This loader targets the HuggingFace mirror `flaviagiammarino/vqa-rad`,
which is distributed as Parquet files (not the original OSF JSON):

    root/
    └── data/
        ├── train-00000-of-00001-*.parquet
        └── test-00000-of-00001-*.parquet

Each Parquet row has three columns:
    image:    {bytes: <raw image bytes>, path: <original filename str>}
    question: str
    answer:   str

Two adaptations are needed to fit the VQASample contract:

1. No `answer_type` column exists in this mirror. We infer it: an answer
   of "yes"/"no" -> "closed", anything else -> "open". This is the
   standard heuristic for this mirror; it is an approximation (a few
   non-yes/no closed answers get labeled "open"), documented in
   _infer_answer_type below.

2. Images are stored as bytes inside the Parquet, but VQASample.image_path
   requires a path to a file on disk. On first load we materialize each
   image to <root>/extracted_images/ and store that path. This keeps the
   bytes-in-Parquet quirk contained inside this loader.
"""

import glob
import io
import json
from pathlib import Path
from typing import Iterator, List

import pyarrow.parquet as pq
from PIL import Image

from .base import MedVQADataset, VQASample


def _infer_answer_type(answer: str) -> str:
    """FALLBACK ONLY: infer 'closed' vs 'open' from the answer string.

    The yes/no heuristic is NOT accurate for VQA-RAD -- the dataset's real
    answer_type depends on the actual answer given (yes/no answer -> closed,
    descriptive answer -> open), which this crude check only partially
    captures. The loader uses real labels from the original VQA-RAD
    distribution via a (question, answer) lookup; this function is only
    invoked if a (question, answer) pair is somehow not found, so the loader
    degrades gracefully instead of crashing.
    """
    if str(answer).strip().lower() in ("yes", "no"):
        return "closed"
    return "open"


def _make_lookup_key(question: str, answer) -> str:
    """Build the normalized (question, answer) lookup key.

    Both fields are str()-coerced (some VQA-RAD answers are numbers),
    stripped, and lowercased so the key matches regardless of casing or
    type differences between the original JSON and the HF parquet mirror.
    The ' ||| ' separator is an arbitrary unlikely-to-occur delimiter.
    """
    q = str(question).strip().lower()
    a = str(answer).strip().lower()
    return q + " ||| " + a


def _load_answer_type_lookup(root: Path) -> dict:
    """Load the (question, answer) -> answer_type lookup built from the
    original VQA-RAD distribution.

    The HuggingFace Parquet mirror dropped the answer_type field; this
    lookup (produced by a one-time script from 'VQA_RAD Dataset Public.json',
    keyed on normalized question+answer) restores the real labels. Returns
    an empty dict if the lookup file is absent, in which case the loader
    falls back to the heuristic for every sample (and warns loudly).
    """
    lookup_path = root / "original" / "answer_type_lookup.json"
    if not lookup_path.exists():
        return {}
    with open(lookup_path) as f:
        return json.load(f)


class VQARadDataset(MedVQADataset):
    """Loader for the HuggingFace Parquet mirror of VQA-RAD."""

    @property
    def name(self) -> str:
        return "vqa_rad"

    def _parquet_path(self) -> str:
        """Find the Parquet file for the requested split.

        The filenames carry a hash suffix (train-00000-of-00001-<hash>.parquet)
        so we glob on the split prefix rather than hard-coding the full name.
        """
        # self.split is "train"/"val"/"test"; this mirror only ships train+test.
        split = self.split
        if split == "val":
            raise ValueError(
                "The VQA-RAD HuggingFace mirror has no validation split "
                "(only train and test). Use --split test for evaluation, "
                "or --split train if you specifically need training data."
            )
        pattern = str(self.root / "data" / f"{split}-*.parquet")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(
                f"No Parquet file matching '{pattern}'. "
                f"Check that --dataset-root points at the VQA-RAD directory "
                f"(it should contain a 'data/' subfolder)."
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Expected exactly one Parquet file for split '{split}', "
                f"found {len(matches)}: {matches}"
            )
        return matches[0]

    def _ensure_images_extracted(self) -> Path:
        """Materialize images from the Parquet to disk, once.

        Returns the directory containing the extracted image files. If the
        directory already exists and has the expected number of files, the
        extraction is skipped (so only the first run pays the cost).
        """
        out_dir = self.root / "extracted_images" / self.split
        out_dir.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(self._parquet_path(), columns=["image"])
        image_col = table.column("image").to_pylist()  # list of {bytes, path} dicts

        # Fast path: if every file is already on disk, skip re-extraction.
        existing = list(out_dir.glob("*"))
        if len(existing) == len(image_col):
            return out_dir

        for idx, img_dict in enumerate(image_col):
            # Each entry is {"bytes": b"...", "path": "xmasy.jpg"} (or similar).
            # We name files by row index so they're stable and collision-free:
            # the original `path` field is not guaranteed unique across rows.
            raw_bytes = img_dict["bytes"]
            # Decode with PIL, then re-save as PNG. Decoding once here means
            # a corrupt image fails now (loudly, with an index) rather than
            # mid-evaluation.
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            image.save(out_dir / f"{idx:05d}.png")

        return out_dir

    def _load_samples(self) -> List[VQASample]:
        """Read the Parquet and build the full list of VQASample objects."""
        parquet_path = self._parquet_path()
        images_dir = self._ensure_images_extracted()

        table = pq.read_table(parquet_path, columns=["question", "answer"])
        questions = table.column("question").to_pylist()
        answers = table.column("answer").to_pylist()

        # Real answer_type labels from the original VQA-RAD distribution,
        # keyed on normalized (question, answer).
        answer_type_lookup = _load_answer_type_lookup(self.root)
        heuristic_fallback_count = 0

        samples: List[VQASample] = []
        for idx, (question, answer) in enumerate(zip(questions, answers)):
            key = _make_lookup_key(question, answer)
            if key in answer_type_lookup:
                answer_type = answer_type_lookup[key]
            else:
                # (question, answer) pair not in the original distribution's
                # lookup -- fall back to the heuristic and count it.
                # KNOWN: as of the current HF mirror + OSF original VQA-RAD,
                # exactly 1 test-split question ("are the borders of the mass
                # well-defined?" / "no") is absent from the original JSON --
                # a minor mismatch between the two dataset distributions. The
                # heuristic labels it 'closed', which is correct for that
                # yes/no question, so the fallback has zero metric impact here.
                answer_type = _infer_answer_type(answer)
                heuristic_fallback_count += 1

            samples.append(
                VQASample(
                    question_id=f"vqa_rad_{self.split}_{idx:05d}",
                    image_path=str(images_dir / f"{idx:05d}.png"),
                    question=question,
                    answer=str(answer),   # normalize to str at the boundary
                    answer_type=answer_type,
                    dataset="vqa_rad",
                    metadata=None,
                )
            )

        if not answer_type_lookup:
            print(f"[vqa_rad] WARNING: no answer_type lookup found at "
                  f"{self.root}/original/answer_type_lookup.json -- "
                  f"using heuristic for ALL {len(samples)} samples.")
        elif heuristic_fallback_count > 0:
            print(f"[vqa_rad] NOTE: {heuristic_fallback_count}/{len(samples)} "
                  f"(question, answer) pairs not found in answer_type lookup "
                  f"-- used heuristic fallback for those.")

        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples

    def __iter__(self) -> Iterator[VQASample]:
        # _load_samples is cheap enough to call here; if iteration happened
        # many times we'd cache it, but the runner iterates exactly once.
        return iter(self._load_samples())

    def __len__(self) -> int:
        return len(self._load_samples())