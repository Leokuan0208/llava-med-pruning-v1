"""Build train_open_answers.json — the candidate set for LLaVA-Med v1.0 open-question
scoring on VQA-RAD.

v1.0's official scoring (run_eval.py, calculate_appearance_with_normalization) does NOT
score the model's literal output. For open questions, it converts the output into a
"predicted label" by checking which training-set answer appears in the output (case-
insensitive substring match), then scores predicted-label == ground-truth.

This script produces that candidate set:
  1. Instantiate VQARadDataset on the TRAIN split. The loader already joins answer_type
     labels via answer_type_lookup.json (Bug #3 fix from May 15), so we don't
     reimplement that here -- we just consume the samples it produces.
  2. Filter to answer_type == "open".
  3. Collect every unique answer, lowercased and stripped.
  4. Write to eval/v1_assets/train_open_answers.json.

Run once; the output is static for the lifetime of VQA-RAD's train split and lives in
git as part of the v1.0 harness.

Usage (from harness root):
    PYTHONPATH=. python eval/build_train_open_answers.py
"""

import json
import random
import sys
from pathlib import Path

from eval.datasets.vqa_rad import VQARadDataset


# Path to the VQA-RAD dataset root (the directory containing data/, extracted_images/,
# original/, etc.). Same path the v1.5 harness uses, mounted into the container by
# KUBERUN.
VQA_RAD_ROOT = Path("/data/dan/dataset/vqa_rad")


def main():
    # === Step 1: Instantiate the loader on the train split ===========================
    # The loader does three things on construction (or first iteration, depending on
    # the parent class): reads the parquet, extracts images to disk if not already
    # there, and joins real answer_type labels from answer_type_lookup.json.
    # We pass split="train" since the candidate set is built from train answers only;
    # test-split answers must NEVER be used to build the candidate set, or evaluation
    # is no longer measuring generalization.
    dataset = VQARadDataset(root=VQA_RAD_ROOT, split="train")

    # The parent class is expected to expose iteration. If `len(dataset)` fails, we
    # fall back to iterating without a length report -- not blocking, just less
    # informative.
    try:
        print(f"[info] loaded train dataset with {len(dataset)} samples")
    except TypeError:
        print(f"[info] loaded train dataset (length not exposed by parent class)")

    # === Step 2: Filter to open questions only =======================================
    # VQASample exposes answer_type as an attribute (per _load_samples). Closed
    # questions are scored by a different v1.0 code path (calculate_exactmatch /
    # yes-no accuracy) and don't need a candidate set.
    open_samples = [s for s in dataset if s.answer_type == "open"]
    print(f"[info] filtered to {len(open_samples)} open train samples")

    # === Step 3: Collect unique normalized answers ===================================
    # Normalization must match what the runner applies to predictions before substring
    # search. Lowercase + strip is what v1.0's calculate_appearance_with_normalization
    # uses. Defensively skip empties: an empty string as a candidate would substring-
    # match every prediction and silently break scoring.
    candidates = set()
    for s in open_samples:
        ans = str(s.answer).strip().lower()
        if ans:
            candidates.add(ans)
    print(f"[info] {len(candidates)} unique open answers in candidate set")

    # Sanity check: v1.0's paper-era VQA-RAD candidate set is ~200-500 entries. Outside
    # 100-800 suggests something fundamental is wrong (mislabeled split, broken join,
    # surprise non-VQA-RAD data in the parquet). Warn loudly, don't crash -- the
    # downstream usage will still surface a real problem on first eval run.
    if not (100 <= len(candidates) <= 800):
        print(
            f"[warn] candidate-set size {len(candidates)} is outside the expected "
            f"~200-500 range. Spot-check the train split and the answer_type join "
            f"before relying on this file.",
            file=sys.stderr,
        )

    # === Step 4: Write to eval/v1_assets/train_open_answers.json =====================
    # Sorted for determinism: the same train split always produces a byte-identical
    # file, which means git diffs only show real changes, never iteration-order churn.
    out_dir = Path(__file__).parent / "v1_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_open_answers.json"

     # v1.0 wraps the candidate list under the key "0" -- matches
    # calculate_appearance_with_normalization's `candidate_set = candidate_set['0']`
    # access pattern in LLaVA-Med-v1.0/llava/eval/eval_metrics/evaluate_metrics.py.
    with out_path.open("w") as f:
        json.dump({"0": sorted(candidates)}, f, indent=2, ensure_ascii=False)

    print(f"[ok] wrote {out_path}")

    # === Sanity preview: 10 random candidates so you can eyeball quality =============
    # Seeded so the preview is reproducible across runs -- useful when comparing two
    # runs of this script (after a dataset update, say).
    sample = random.Random(42).sample(sorted(candidates), min(10, len(candidates)))
    print(f"[info] sample candidates (seeded): {sample}")


if __name__ == "__main__":
    main()