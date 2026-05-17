"""3-sample smoke test for LLaVA-Med v1.0 inference path.

Loads the merged VQA-RAD model, runs inference on 3 VQA-RAD test samples,
and prints (question, predicted answer, ground truth). Should take 1-2
minutes total -- if it hangs much longer, something is wrong with model
loading or generation.
"""

import sys
from pathlib import Path

# Ensure harness root is on PYTHONPATH if not already
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.model_loader import load_llava_med_v1
from eval.runner import Runner
from eval.datasets.vqa_rad import VQARadDataset


MODEL_PATH = "/data/dan/weights/llava-med-7b-vqarad-merged"
DATASET_ROOT = "/data/dan/dataset/vqa_rad"


def main():
    print("[1/3] Loading model... (~30-60s for a 13GB fp16 load)")
    loaded = load_llava_med_v1(MODEL_PATH)
    print(f"      conv_mode={loaded.conv_mode} "
          f"image_token_len={loaded.image_token_len} "
          f"mm_use_im_start_end={loaded.mm_use_im_start_end}")

    print("[2/3] Loading 3 VQA-RAD test samples...")
    dataset = VQARadDataset(root=DATASET_ROOT, split="test")
    samples = list(dataset)[:3]
    for s in samples:
        print(f"      {s.question_id}: [{s.answer_type}] {s.question!r} -> {s.answer!r}")

    print("[3/3] Running inference on 3 samples...")
    runner = Runner(loaded)
    predictions = runner.run(samples, progress=False)

    print()
    print("=" * 60)
    print("Smoke test results:")
    print("=" * 60)
    for sample, pred in zip(samples, predictions):
        print(f"\n[{sample.question_id}] type={sample.answer_type}")
        print(f"  Q: {sample.question}")
        print(f"  GT: {sample.answer}")
        print(f"  Pred: {pred.text}")
        print(f"  Latency: {pred.latency_ms:.0f}ms | "
              f"in_tok={pred.n_input_tokens} out_tok={pred.n_output_tokens}")


if __name__ == "__main__":
    main()