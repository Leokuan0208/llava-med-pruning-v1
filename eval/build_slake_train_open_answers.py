"""Build slake_train_open_answers.json -- candidate set for LLaVA-Med v1.0
open-question scoring on SLAKE (English subset).

Mirrors build_vqa_rad_train_open_answers.py and build_path_vqa_train_open_answers.py:
collect every unique open-question answer from SLAKE's English train split,
normalize (lowercase + strip), write as v1.0's expected {"0": [...]} wrapped
JSON. Score-time substring-and-argmax matching consumes this exact format.

The SLAKE loader already filters to q_lang == 'en' inside, so we just iterate
the train split and pull open-question answers.

Usage (from harness root):
    PYTHONPATH=. python eval/build_slake_train_open_answers.py
"""

import json
from pathlib import Path

from eval.datasets.slake import SlakeDataset


SLAKE_ROOT = Path("/data/dan/dataset/slake")
OUTPUT_FILE = Path(__file__).parent / "v1_assets" / "slake_train_open_answers.json"


def main():
    print(f"Loading SLAKE train split from {SLAKE_ROOT}...")
    ds = SlakeDataset(root=str(SLAKE_ROOT), split="train")
    samples = list(ds)
    print(f"  Total English train samples: {len(samples)}")

    # Collect unique open-question answers, normalised.
    open_answers = set()
    for s in samples:
        if s.answer_type == "open":
            normalized = s.answer.strip().lower()
            if normalized:
                open_answers.add(normalized)

    candidates = sorted(open_answers)
    print(f"  Unique open answers: {len(candidates)}")
    print(f"  Example candidates: {candidates[:10]}")

    # v1.0 expects the wrapped {"0": [...]} format.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"0": candidates}, f, indent=2)
    print(f"Wrote {len(candidates)} candidates to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
