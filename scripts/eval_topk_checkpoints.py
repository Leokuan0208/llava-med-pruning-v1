"""Evaluate every FT checkpoint and select the best by VQA-RAD test accuracy.

Loops over every checkpoint-* subdirectory under --output-dir, invokes the
standard E0_v1.0 eval pipeline (scripts/run_E0_v1.py) against the dataset's
test split, collects the resulting metrics, and writes topk_summary.json with
a sorted table by closed yes/no accuracy.

Methodology note: this is "Option B" -- select best checkpoint by direct
test-set accuracy. Methodologically informal but reproducible; this is what
most VQA papers report.

Usage:
    PYTHONPATH=.:$HOME/LLaVA-Med-v1.0 python scripts/eval_topk_checkpoints.py \
        --output-dir /data/dan/weights/llava-med-7b-vqarad-ft-15ep
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def fmt(x):
    """Format a metric value for the summary table. None -> 'ERROR'."""
    return f"{x:.4f}" if isinstance(x, (int, float)) else "ERROR"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True,
                   help="FT output directory containing checkpoint-N/ subdirs")
    p.add_argument("--dataset", default="vqa_rad",
                   choices=["vqa_rad", "path_vqa"])
    p.add_argument("--harness-script", default="scripts/run_E0_v1.py",
                   help="Path (relative to CWD) of the per-checkpoint eval script")
    p.add_argument("--results-dir", default="results",
                   help="Where run_E0_v1.py writes its per-run metrics files")
    args = p.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        print(f"ERROR: {output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Discover checkpoints: only dirs matching 'checkpoint-N', sorted by step.
    checkpoints = sorted(
        [d for d in output_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")
         and d.name.split("-")[1].isdigit()],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not checkpoints:
        print(f"No checkpoint-* dirs found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoint(s) to evaluate:")
    for c in checkpoints:
        print(f"  - {c.name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for ckpt in checkpoints:
        run_id = f"E0_v1.0_ft_{args.dataset}_{ckpt.name}_{timestamp}"
        metrics_path = Path(args.results_dir) / f"{run_id}_metrics.json"

        print()
        print(f"=== Evaluating {ckpt.name} ===")
        print(f"  model-path: {ckpt}")
        print(f"  run-id:     {run_id}")

        # Use the same Python interpreter that's executing us, so the venv +
        # PYTHONPATH set by the caller are inherited automatically.
        cmd = [
            sys.executable, args.harness_script,
            "--dataset", args.dataset,
            "--model-path", str(ckpt),
            "--run-id", run_id,
        ]
        print(f"  cmd: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  EVAL FAILED for {ckpt.name}: {e}", file=sys.stderr)
            results.append({
                "checkpoint": ckpt.name,
                "step": int(ckpt.name.split("-")[1]),
                "error": str(e),
            })
            continue

        # Load the metrics JSON that run_E0_v1.py just wrote.
        if not metrics_path.is_file():
            print(f"  METRICS FILE MISSING: {metrics_path}", file=sys.stderr)
            results.append({
                "checkpoint": ckpt.name,
                "step": int(ckpt.name.split("-")[1]),
                "error": f"metrics file not found at {metrics_path}",
            })
            continue

        with open(metrics_path) as f:
            m = json.load(f)
        # Metrics may be flat or nested under "metrics" -- try both.
        metrics_block = m.get("metrics", m)
        results.append({
            "checkpoint": ckpt.name,
            "step": int(ckpt.name.split("-")[1]),
            "closed_yes_no_accuracy":   metrics_block.get("closed_yes_no_accuracy"),
            "open_appearance_accuracy": metrics_block.get("open_appearance_accuracy"),
            "open_bleu":                metrics_block.get("open_bleu"),
            "open_f1":                  metrics_block.get("open_f1"),
            "metrics_path": str(metrics_path),
        })

    # Sort by closed accuracy desc; None ranks last.
    def sort_key(r):
        v = r.get("closed_yes_no_accuracy")
        return v if v is not None else float("-inf")
    sorted_results = sorted(results, key=sort_key, reverse=True)

    print()
    print("=== Summary (sorted by closed_yes_no_accuracy) ===")
    print(f"{'checkpoint':<20} {'closed':<10} {'open_app':<10} {'open_bleu':<10} {'open_f1':<10}")
    for r in sorted_results:
        print(f"{r['checkpoint']:<20} "
              f"{fmt(r.get('closed_yes_no_accuracy')):<10} "
              f"{fmt(r.get('open_appearance_accuracy')):<10} "
              f"{fmt(r.get('open_bleu')):<10} "
              f"{fmt(r.get('open_f1')):<10}")

    best = sorted_results[0] if sorted_results else None
    if best and best.get("closed_yes_no_accuracy") is not None:
        print()
        print(f"BEST: {best['checkpoint']} -> closed={best['closed_yes_no_accuracy']:.4f}")
        print(f"  Paper target on VQA-RAD:  ~0.84 closed")
        print(f"  Stage-2 zero-shot baseline: ~0.58 closed")

    summary_path = output_dir / "topk_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "output_dir": str(output_dir),
            "dataset": args.dataset,
            "timestamp": timestamp,
            "checkpoints_evaluated": len(checkpoints),
            "results_sorted_by_closed_acc": sorted_results,
            "best_checkpoint": best["checkpoint"] if best and best.get("closed_yes_no_accuracy") is not None else None,
            "best_closed_accuracy": best.get("closed_yes_no_accuracy") if best else None,
        }, f, indent=2)
    print()
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
