"""PathVQA dataset loader.

PathVQA (He et al., 2020): ~32,000 question-answer pairs over ~5,000
pathology images. Mix of yes/no (closed) and free-form (open) questions
by design -- the dataset was constructed with a roughly 50/50 split.

This loader targets the HuggingFace mirror `flaviagiammarino/path-vqa`,
which is distributed as Parquet files. Unlike the VQA-RAD mirror, the
PathVQA mirror splits each set across multiple shards:

    root/
    └── data/
        ├── train-00000-of-00007-*.parquet
        ├── train-00001-of-00007-*.parquet
        ├── ...
        ├── validation-00000-of-00003-*.parquet
        ├── ...
        ├── test-00000-of-00003-*.parquet
        ├── test-00001-of-00003-*.parquet
        └── test-00002-of-00003-*.parquet

Each Parquet row has three columns (same schema as VQA-RAD's mirror):
    image:    {bytes: <raw image bytes>, path: <original filename str>}
    question: str
    answer:   str

Two adaptations are needed to fit the VQASample contract, both identical
to the VQA-RAD adaptations:

1. No answer_type column exists in this mirror. We infer it: an answer
   of "yes"/"no" -> "closed", anything else -> "open". This is a
   well-established heuristic for this mirror, and PathVQA's closed
   questions are essentially all yes/no by construction, so the
   mislabel rate is expected to be very low (unlike VQA-RAD, which had
   "which side?" -> "left" style closed questions that the heuristic
   misses). A future authoritative join against the original pickle
   distribution remains an option if exact labels are ever needed.

2. Images are stored as bytes inside the Parquet, but VQASample.image_path
   requires a path to a file on disk. On first load we materialize each
   image to <root>/extracted_images/<split>/ and store that path. Same
   trick as the VQA-RAD loader.

Multi-shard handling:
    Shards are loaded in sorted order (lexicographic = numeric because
    of zero-padded names). Row indexing is GLOBAL across shards, so
    question_ids and image filenames are sequential 0..N-1 regardless
    of which shard a row originally came from. Internal sharding is
    a quirk of the mirror, not exposed to downstream code.
"""

import glob
import io
from pathlib import Path
from typing import Iterator, List

import pyarrow.parquet as pq
from PIL import Image

from .base import MedVQADataset, VQASample


# Expected row counts per split (from authoritative HuggingFace dataset
# card, verified May 13 2026). Used as a safety check after loading.
_EXPECTED_ROWS = {
    "train": 19654,
    "val": 6259,
    "validation": 6259,
    "test": 6719,
}


def _infer_answer_type(answer: str) -> str:
    """Infer 'closed' vs 'open' from the answer string.

    Same yes/no heuristic as the VQA-RAD loader. Documented as an
    approximation: a small number of non-yes/no closed answers may be
    mislabelled as 'open'. PathVQA's closed questions are almost all
    yes/no by construction so this mislabel rate is expected to be
    very low, but it is non-zero.
    """
    if answer.strip().lower() in ("yes", "no"):
        return "closed"
    return "open"


class PathVQADataset(MedVQADataset):
    """Loader for the HuggingFace Parquet mirror of PathVQA (sharded)."""

    @property
    def name(self) -> str:
        return "path_vqa"

    def _split_filename_prefix(self) -> str:
        """Translate our contract's split name to PathVQA's filename prefix.

        PathVQA uses 'validation' instead of 'val' in its filenames; we
        accept both for caller convenience.
        """
        return {"val": "validation"}.get(self.split, self.split)

    def _parquet_paths(self) -> List[str]:
        """Find all Parquet shards for the requested split, sorted.

        Sorted is critical: zero-padded shard numbers (00000, 00001, ...)
        sort lexicographically to match numerical order, so row indices
        across the concatenated dataset stay stable across runs.
        """
        if self.split not in _EXPECTED_ROWS:
            raise ValueError(
                f"Unknown split {self.split!r}. Expected one of "
                f"{sorted(_EXPECTED_ROWS)}."
            )
        prefix = self._split_filename_prefix()
        pattern = str(self.root / "data" / f"{prefix}-*.parquet")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No Parquet files matching '{pattern}'. "
                f"Check that --dataset-root points at the PathVQA directory "
                f"(it should contain a 'data/' subfolder)."
            )
        return matches

    def _ensure_images_extracted(self) -> Path:
        """Materialize images from all Parquet shards to disk, once.

        Images are named by their GLOBAL row index across the concatenated
        dataset, so 00000.png..06718.png for test. This matches the
        question_id numbering and means a question_id maps to its image
        file with no shard arithmetic.

        Returns the directory containing the extracted images. If the
        directory already exists and has the expected number of files,
        re-extraction is skipped (so only the first run pays the cost).
        """
        out_dir = self.root / "extracted_images" / self.split
        out_dir.mkdir(parents=True, exist_ok=True)

        expected_n = _EXPECTED_ROWS[self.split]
        existing = list(out_dir.glob("*.png"))
        if len(existing) == expected_n:
            return out_dir

        # Fresh (or partial) extraction. Walk each shard in order and
        # emit images with a global row index.
        global_idx = 0
        for shard_path in self._parquet_paths():
            table = pq.read_table(shard_path, columns=["image"])
            image_col = table.column("image").to_pylist()
            for img_dict in image_col:
                # Each entry is {"bytes": b"...", "path": "..."}
                raw_bytes = img_dict["bytes"]
                # Decode once now so corrupt images fail loudly with an
                # index rather than mid-evaluation.
                image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                image.save(out_dir / f"{global_idx:05d}.png")
                global_idx += 1

        # Post-extraction safety check: did we get exactly the expected
        # number of files? A mismatch means either expected_n is wrong or
        # a shard had unexpected size -- either way, fail loud.
        actual_n = len(list(out_dir.glob("*.png")))
        if actual_n != expected_n:
            raise RuntimeError(
                f"After extraction, found {actual_n} images in {out_dir}, "
                f"expected {expected_n}. Check shard integrity."
            )
        return out_dir

    def _load_samples(self) -> List[VQASample]:
        """Read all shards, build the full list of VQASample objects.

        Done eagerly: the PathVQA test split is 6,719 rows -- under 10 MB
        as dataclasses in memory.  Eager loading makes len() exact and
        keeps the pattern consistent with the other loaders.
        """
        shard_paths = self._parquet_paths()
        images_dir = self._ensure_images_extracted()

        samples: List[VQASample] = []
        global_idx = 0
        for shard_path in shard_paths:
            # Skip the image column on this read -- we already have images
            # on disk from _ensure_images_extracted. Only read what we need.
            table = pq.read_table(shard_path, columns=["question", "answer"])
            questions = table.column("question").to_pylist()
            answers = table.column("answer").to_pylist()

            for question, answer in zip(questions, answers):
                samples.append(
                    VQASample(
                        question_id=f"path_vqa_{self.split}_{global_idx:05d}",
                        image_path=str(images_dir / f"{global_idx:05d}.png"),
                        question=question,
                        answer=str(answer),
                        answer_type=_infer_answer_type(answer),
                        dataset="path_vqa",
                        metadata=None,
                    )
                )
                global_idx += 1

        # Safety check: total row count should match the expected count.
        # If this fails, either a shard is shorter than expected or
        # _EXPECTED_ROWS is stale; either way it warrants attention.
        expected_n = _EXPECTED_ROWS[self.split]
        if len(samples) != expected_n:
            raise RuntimeError(
                f"After loading all shards, got {len(samples)} samples, "
                f"expected {expected_n}. Check shard integrity."
            )

        if self.max_samples is not None:
            samples = samples[: self.max_samples]

        return samples

    def __iter__(self) -> Iterator[VQASample]:
        return iter(self._load_samples())

    def __len__(self) -> int:
        return len(self._load_samples())