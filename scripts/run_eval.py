#!/usr/bin/env python
"""CLI entry point for evaluation runs.

Example invocations:

    # Baseline on VQA-RAD
    python scripts/run_eval.py \\
        --model-path /data/dan/weights/llava-med-v1.5-mistral-7b \\
        --dataset vqa_rad \\
        --dataset-root /data/dan/dataset/vqa_rad \\
        --method baseline \\
        --output-dir results/

    # Smoke test on the first 20 questions
    python scripts/run_eval.py \\
        --model-path /data/dan/weights/llava-med-v1.5-mistral-7b \\
        --dataset vqa_rad \\
        --dataset-root /data/dan/dataset/vqa_rad \\
        --method baseline \\
        --max-samples 20 \\
        --output-dir results/

Outputs two files per run in --output-dir:
    {run_id}_metrics.json     - aggregate metrics + resolved config
    {run_id}_predictions.jsonl - per-question predictions
where run_id = "{method}_{dataset}_{keep_ratio}_{timestamp}".
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to the import path so 'eval' resolves correctly
# regardless of where the script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.model_loader import load_llava_med
from eval.runner import run_eval
from eval.datasets.vqa_rad import VQARadDataset
from eval.datasets.slake import SlakeDataset
from eval.datasets.path_vqa import PathVQADataset
from eval.methods.baseline import BaselineMethod


# Maps the --dataset choice string to its loader class. Adding a new
# benchmark later is a one-line addition here.
_DATASET_REGISTRY = {
    "vqa_rad": VQARadDataset,
    "slake": SlakeDataset,
    "path_vqa": PathVQADataset,
}

# Maps the --method choice string to its class. Grows as methods are
# implemented; for now only the no-op baseline exists.
_METHOD_REGISTRY = {
    "baseline": BaselineMethod,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model
    p.add_argument("--model-path", required=True,
                   help="Path to LLaVA-Med v1.5 model directory.")
    p.add_argument("--device", default="cuda",
                   help="Torch device. Default: cuda.")

    # Dataset
    p.add_argument("--dataset", required=True,
                   choices=["vqa_rad", "slake", "path_vqa"],
                   help="Which benchmark to evaluate on.")
    p.add_argument("--dataset-root", required=True,
                   help="Path to the dataset's root directory on disk.")
    p.add_argument("--split", default="test",
                   help="Dataset split: train/val/test. Default: test.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="If set, evaluate only the first N samples "
                        "(useful for smoke tests).")

    # Method
    p.add_argument("--method", required=True,
                   choices=["baseline"],  # more added as implemented
                   help="Pruning method to apply.")
    p.add_argument("--keep-ratio", type=float, default=1.0,
                   help="Fraction of visual tokens to keep. Ignored "
                        "by the baseline method.")
    p.add_argument("--layer-k", type=int, default=2,
                   help="LLM layer at which to apply pruning. Ignored "
                        "by the baseline method.")

    # Generation
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="Generation length cap.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature. 0.0 = greedy.")

    # Output
    p.add_argument("--output-dir", default="results/",
                   help="Directory to write metrics and predictions files.")
    p.add_argument("--run-id", default=None,
                   help="Override the auto-generated run ID.")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def _build_run_id(args) -> str:
    """Construct the run identifier used to name both output files.

    Format: {method}_{dataset}_{keep_ratio}_{timestamp}
    e.g.    baseline_vqa_rad_kr1.0_20260514-091500

    The keep_ratio is in the name even though the baseline ignores it,
    so that later pruning runs (which DO use it) sort and compare
    cleanly alongside the baseline. The timestamp prevents a re-run
    from silently overwriting a previous run's files.
    """
    if args.run_id is not None:
        return args.run_id
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # "kr" = keep ratio; formatted to one decimal so 1.0 -> "kr1.0".
    return f"{args.method}_{args.dataset}_kr{args.keep_ratio:.1f}_{timestamp}"


def main():
    args = parse_args()

    # --- 1. Resolve the dataset and method classes from the registries -----
    # argparse's `choices=` already guarantees the keys exist, so no
    # KeyError guard is needed here.
    dataset_cls = _DATASET_REGISTRY[args.dataset]
    method_cls = _METHOD_REGISTRY[args.method]

    # --- 2. Build the dataset ----------------------------------------------
    print(f"[run_eval] Building dataset '{args.dataset}' "
          f"(split={args.split}, max_samples={args.max_samples}) ...")
    dataset = dataset_cls(
        root=args.dataset_root,
        split=args.split,
        max_samples=args.max_samples,
    )
    print(f"[run_eval] Dataset ready: {len(dataset)} samples.")

    # --- 3. Instantiate the pruning method ---------------------------------
    # keep_ratio and layer_k are passed as kwargs; PruningMethod.__init__
    # stashes them in self.config. BaselineMethod ignores them, but real
    # methods will read them -- passing them uniformly means the CLI does
    # not need per-method special-casing.
    method = method_cls(keep_ratio=args.keep_ratio, layer_k=args.layer_k)
    print(f"[run_eval] Method: {method.name}")

    # --- 4. Load the model -------------------------------------------------
    # This is the slow step (~minutes, 15 GB of weights).
    print(f"[run_eval] Loading model from {args.model_path} ...")
    loaded_model = load_llava_med(args.model_path, device=args.device)
    print(f"[run_eval] Model loaded (conv_mode={loaded_model.conv_mode}).")

    # --- 5. Run the evaluation ---------------------------------------------
    print(f"[run_eval] Running evaluation ...")
    result = run_eval(
        loaded_model=loaded_model,
        method=method,
        dataset=dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    # --- 6. Write the two output files -------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _build_run_id(args)

    # 6a. Metrics JSON: aggregate numbers + the full resolved config.
    # vars(args) captures every CLI argument as a dict, so the run is
    # fully reproducible from this file alone.
    metrics_path = output_dir / f"{run_id}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "metrics": result["metrics"],
                "config": result["config"],
                "args": vars(args),
            },
            f,
            indent=2,
        )

    # 6b. Predictions JSONL: one JSON object per line, one line per
    # question. JSONL (not JSON) so it can be streamed/grepped and so a
    # single malformed line does not break the whole file.
    predictions_path = output_dir / f"{run_id}_predictions.jsonl"
    with open(predictions_path, "w") as f:
        for pred in result["predictions"]:
            f.write(json.dumps(pred) + "\n")

    # --- 7. Print a summary to the terminal --------------------------------
    m = result["metrics"]
    print("\n" + "=" * 60)
    print(f"[run_eval] Run complete: {run_id}")
    print("=" * 60)
    print(f"  samples         : {m['n_total']} "
          f"({m['n_closed']} closed, {m['n_open']} open)")
    print(f"  closed accuracy : {m['closed_accuracy']}")
    print(f"  open recall     : {m['open_recall']}")
    print(f"  overall accuracy: {m['overall_accuracy']}")
    print(f"  mean latency    : {m['mean_latency_ms']} ms")
    print(f"  peak GPU memory : {m['peak_gpu_memory_gb']:.2f} GiB")
    print("=" * 60)
    print(f"  metrics    -> {metrics_path}")
    print(f"  predictions-> {predictions_path}")


if __name__ == "__main__":
    main()