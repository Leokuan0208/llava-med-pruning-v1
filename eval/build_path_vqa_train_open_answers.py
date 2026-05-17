"""Build path_vqa_train_open_answers.json -- candidate set for LLaVA-Med
v1.0 open-question scoring on PathVQA.

Same shape and intent as build_train_open_answers.py (the VQA-RAD version):
collect every unique open-question answer from PathVQA's TRAIN split,
normalize (lowercase + strip), write as v1.0's expected {"0": [...]} wrapped
JSON. Score-time substring-and-argmax matching consumes this exact format.

Differences from the VQA-RAD builder:

  - Uses PathVQADataset instead of VQARadDataset.
  - Output filename is path_vqa_train_open_answers.json (distinct from
    VQA-RAD's train_open_answers.json) so both files can coexist.
  - answer_type filter still uses lowercase 'open' since the loader emits
    that case.
  - The expected candidate count is much larger than VQA-RAD's 402 --
    PathVQA's train split has ~20k samples vs VQA-RAD's 1.8k. Open
    questions are roughly half, so we expect ~3000-6000 unique answers.
    Sanity check band widened accordingly.

Run once; the output is static for the lifetime of PathVQA's train split
and should live in git as part of the v1.0 harness.

Usage (from harness root):
    PYTHONPATH=. python eval/build_path_vqa_train_open_answers.py
"""

import json
import random
import sys
from pathlib import Path

from eval.datasets.path_vqa import PathVQADataset


# Path to the PathVQA dataset root. Same path the v1.0 harness uses
# everywhere; the value lives in run_E0_v1.py's DATASET_REGISTRY as well.
PATH_VQA_ROOT = Path("/data/dan/dataset/path_vqa")


def main():
    # === Step 1: Instantiate the loader on the train split =====================
    # PathVQADataset's __init__ takes (root, split, max_samples=None).
    # On first run, this triggers image extraction for the train split
    # (~20k images), which can take several minutes. Subsequent runs see
    # the cached extracted_images/train/ directory and skip re-extraction.
    dataset = PathVQADataset(root=PATH_VQA_ROOT, split="train")

    try:
        print(f"[info] loaded train dataset with {len(dataset)} samples")
    except TypeError:
        print(f"[info] loaded train dataset (length not exposed)")

    # === Step 2: Filter to open questions only =================================
    # The PathVQA loader uses the yes/no heuristic for answer_type (no
    # answer_type column exists in the HF mirror); answers that are NOT
    # 'yes'/'no' get labelled 'open'. By construction this filter pulls
    # exactly the free-form questions, which is what we need for the
    # candidate set.
    open_samples = [s for s in dataset if s.answer_type == "open"]
    print(f"[info] filtered to {len(open_samples)} open train samples")

    # === Step 3: Collect unique normalized answers =============================
    # Normalize identically to VQA-RAD's builder (lowercase + strip). v1.0's
    # calculate_appearance_with_normalization will re-apply normalize_word
    # at score time, so the casing-and-whitespace shape here just needs to
    # be consistent. Empty strings are dropped defensively (they would
    # substring-match every prediction).
    candidates = set()
    for s in open_samples:
        ans = str(s.answer).strip().lower()
        if ans:
            candidates.add(ans)
    print(f"[info] {len(candidates)} unique open answers in candidate set")

    # Sanity check: PathVQA train open answers, expected band ~3k-8k. Way
    # outside that suggests something fundamental is off (wrong split, broken
    # loader, etc.) -- warn loudly but don't crash, since downstream eval
    # will surface real problems anyway.
    if not (2000 <= len(candidates) <= 12000):
        print(
            f"[warn] candidate-set size {len(candidates)} is outside the expected "
            f"~3000-8000 range. Spot-check the train split and the answer_type "
            f"heuristic before relying on this file.",
            file=sys.stderr,
        )

    # === Step 4: Write to eval/v1_assets/path_vqa_train_open_answers.json =====
    # Sorted for determinism: same train split always produces a byte-
    # identical file, which means git diffs only show real changes. Wrapped
    # as {"0": [...]} for v1.0 compat (calculate_appearance_with_normalization
    # accesses candidate_set['0']).
    out_dir = Path(__file__).parent / "v1_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "path_vqa_train_open_answers.json"

    with out_path.open("w") as f:
        json.dump({"0": sorted(candidates)}, f, indent=2, ensure_ascii=False)

    print(f"[ok] wrote {out_path}")

    # === Sanity preview: 10 seeded random candidates so you can eyeball =======
    # Seeded for reproducibility (different seed than VQA-RAD's so preview
    # shows different candidates if the script outputs are eyeballed side-
    # by-side).
    sample = random.Random(2026).sample(sorted(candidates), min(10, len(candidates)))
    print(f"[info] sample candidates (seeded): {sample}")


if __name__ == "__main__":
    main()