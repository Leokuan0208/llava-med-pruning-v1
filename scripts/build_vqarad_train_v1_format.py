"""Build vqa_rad_train_v1_format.json for full fine-tuning.

Converts the VQA-RAD HuggingFace parquet train split into v1.0's expected
JSON training-data format. Reuses extracted_images/train/ (cached by yesterday
or by a prior loader call).

v1.0's train.py expects each item shaped like:
    {
      "id": "vqa_rad_train_00000",
      "image": "00000.png",          # relative to --image_folder
      "conversations": [
        {"from": "human", "value": "<image>\\nquestion text"},
        {"from": "gpt",   "value": "answer text"}
      ]
    }

Notes:
  - The "<image>\\n" prefix tells v1.0's preprocessing where to inject the
    <im_patch> tokens. Don't drop it; the model trains incoherently without.
  - We include ALL train samples (open + closed). v1.0's training was over
    full train splits, not just open or just closed.
  - 'image' is the BASENAME (00000.png), not a full path. v1.0's loader
    joins with --image_folder at training time.

Usage:
    PYTHONPATH=. python scripts/build_vqarad_train_v1_format.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.datasets.vqa_rad import VQARadDataset


VQA_RAD_ROOT = Path("/data/dan/dataset/vqa_rad")
OUTPUT_JSON = VQA_RAD_ROOT / "vqa_rad_train_v1_format.json"


def main():
    print(f"[1/3] Loading VQA-RAD train split via the harness loader...")
    dataset = VQARadDataset(root=VQA_RAD_ROOT, split="train")
    samples = list(dataset)
    print(f"      loaded {len(samples)} train samples")

    print(f"[2/3] Converting to v1.0 conversation format...")
    out = []
    for sample in samples:
        # sample.image_path looks like /data/.../extracted_images/train/00000.png
        # We just want the basename "00000.png" -- v1.0 joins with --image_folder.
        image_basename = Path(sample.image_path).name
        out.append({
            "id": sample.question_id,
            "image": image_basename,
            "conversations": [
                {"from": "human", "value": f"<image>\n{sample.question}"},
                {"from": "gpt",   "value": sample.answer},
            ],
        })
    print(f"      converted {len(out)} samples")

    print(f"[3/3] Writing {OUTPUT_JSON}...")
    with OUTPUT_JSON.open("w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    size_mb = OUTPUT_JSON.stat().st_size / (1024 * 1024)
    print(f"      wrote {OUTPUT_JSON} ({size_mb:.1f} MB)")

    print()
    print(f"=== Sample item ===")
    print(json.dumps(out[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()