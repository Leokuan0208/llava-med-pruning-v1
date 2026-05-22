"""E0_v1.0: zero/few-shot baseline for LLaVA-Med v1.0, multi-dataset.

Runs the v1.0 baseline on VQA-RAD, PathVQA, or SLAKE test sets.
Loads a merged LLaVA-Med v1.0 checkpoint, runs inference, scores with
v1.0's published method, and writes results + per-question records to disk.

Expected outputs:
  results/E0_v1.0_<dataset>_<split>_<timestamp>_metrics.json     -- aggregate metrics
  results/E0_v1.0_<dataset>_<split>_<timestamp>_predictions.jsonl -- per-question records

Expected wall time on A100:
  - VQA-RAD test (451 samples):  ~2-3 min  (paper closed ~0.84, open appearance ~0.62)
  - PathVQA test (6,719 samples): ~30-50 min (paper closed ~0.91, open appearance ~0.39)

If --skip-open-scoring is set (or the candidate file is missing), open-question
metrics are reported as zeros (inference still runs but scoring is skipped).
Useful for datasets where the per-dataset candidate set hasn't been built yet
-- closed yes/no scoring still works correctly.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Harness root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.model_loader import load_llava_med_v1
from eval.runner import Runner
from eval.metrics import score_predictions
from eval.datasets.vqa_rad import VQARadDataset
from eval.datasets.path_vqa import PathVQADataset
from eval.datasets.slake import SlakeDataset
from eval.methods.baseline import BaselineMethod
from eval.methods.random_pruning import RandomPruning
from eval.methods.question_similarity_pruning import QuestionSimilarityPruning


METHOD_REGISTRY = {
    "baseline": lambda kr: BaselineMethod(),
    "random":   lambda kr: RandomPruning(keep_ratio=kr),
    "qsim":     lambda kr: QuestionSimilarityPruning(keep_ratio=kr),
}


# === Registry-driven dataset configuration =====================================
# Each entry maps a --dataset CLI value to its loader class, default data root,
# and default candidate file. Adding a new dataset is a one-line edit here:
# (a) add the dataset to the registry, (b) ensure the corresponding loader
# class and candidate file exist. Nothing else in this script needs to change.
#
# Why a single registry instead of three separate dicts: keeping (loader, data
# root, candidate file) together makes it impossible to mismatch them. E.g.
# you can't accidentally use VQA-RAD's candidate file for PathVQA scoring,
# because the wrong (dataset, candidate_file) pair simply isn't in the registry.

_V1_ASSETS = Path(__file__).resolve().parent.parent / "eval" / "v1_assets"

DATASET_REGISTRY = {
    "vqa_rad": {
        "loader":         VQARadDataset,
        "data_root":      "/data/dan/dataset/vqa_rad",
        "candidate_file": str(_V1_ASSETS / "vqa_rad_train_open_answers.json"),
    },
    "path_vqa": {
        "loader":         PathVQADataset,
        "data_root":      "/data/dan/dataset/path_vqa",
        "candidate_file": str(_V1_ASSETS / "path_vqa_train_open_answers.json"),
    },
    "slake": {  
        "loader":         SlakeDataset,
        "data_root":      "/data/dan/dataset/slake",
        "candidate_file": str(_V1_ASSETS / "slake_train_open_answers.json"),
    },
}

DEFAULT_MODEL_PATH = "/data/dan/weights/llava-med-7b-vqarad-merged"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args():
    p = argparse.ArgumentParser(description="E0_v1.0: zero/few-shot baseline for LLaVA-Med v1.0")
    p.add_argument("--dataset", default="vqa_rad", choices=sorted(DATASET_REGISTRY.keys()),
                   help="Which dataset to evaluate against")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                   help="Path to the merged LLaVA-Med v1.0 model directory")
    p.add_argument("--dataset-root", default=None,
                   help="Path to the dataset root. If omitted, uses the default for --dataset")
    p.add_argument("--candidate-file", default=None,
                   help="Path to <dataset>_train_open_answers.json. If omitted, uses the "
                        "default for --dataset (ignored if --skip-open-scoring)")
    p.add_argument("--skip-open-scoring", action="store_true",
                   help="Run inference on open questions but report open metrics as zero. "
                        "Useful for datasets where a per-dataset candidate set hasn't been "
                        "built yet -- closed yes/no scoring still works.")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                   help="Where to write metrics + predictions")
    p.add_argument("--split", default="test", choices=["train", "test"],
                   help="Which dataset split to evaluate")
    p.add_argument("--max-samples", type=int, default=None,
                   help="If set, only run the first N samples (for debugging)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for stochastic decoding")
    p.add_argument("--run-id", default=None,
                   help="Override the auto-generated run ID (used in output filenames)")
    p.add_argument("--method", default="baseline", choices=sorted(METHOD_REGISTRY.keys()),
                   help="Pruning method to apply. baseline = no pruning.")
    p.add_argument("--keep-ratio", type=float, default=1.0,
                   help="Fraction of visual tokens to keep (0, 1]. Ignored for baseline.")

    args = p.parse_args()

    # Resolve dataset-driven defaults from the registry.
    registry_entry = DATASET_REGISTRY[args.dataset]
    if args.dataset_root is None:
        args.dataset_root = registry_entry["data_root"]
    if args.candidate_file is None:
        args.candidate_file = registry_entry["candidate_file"]

    # Auto-fallback: if the candidate file doesn't exist on disk and the caller
    # didn't ask to skip open scoring, silently switch to skip mode rather than
    # crashing later. Print a clear notice so it's not silent-silent. This
    # prevents the harness from being blocked on a missing candidate file when
    # the caller really just wants closed yes/no diagnostics.
    if not args.skip_open_scoring and not Path(args.candidate_file).exists():
        print(f"[notice] candidate file not found at {args.candidate_file}; "
              f"auto-enabling --skip-open-scoring. Build it with "
              f"`PYTHONPATH=. python eval/build_{args.dataset}_train_open_answers.py` "
              f"to get full open-question metrics.", file=sys.stderr)
        args.skip_open_scoring = True

    return args


def main():
    args = parse_args()

    # Instantiate the pruning method early so its `.name` is available
    # for the auto-generated run_id below. The method itself doesn't
    # touch the model until runner.run() calls .attach() much later.
    method = METHOD_REGISTRY[args.method](args.keep_ratio)

    # Build a run identifier embedding model + dataset + split + timestamp.
    # The method name (e.g. "baseline", "qsim_kr0p50") is in the run_id
    # so different methods/ratios don't clobber each other's outputs.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"E0_v1.0_{args.dataset}_{method.name}_{args.split}_{timestamp}"

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / f"{run_id}_metrics.json"
    preds_path = results_dir / f"{run_id}_predictions.jsonl"
    log_path = results_dir / f"{run_id}.log"

    # Open the log file alongside stdout. We don't redirect because we want
    # the terminal to also show progress; instead we write key milestones.
    log_f = log_path.open("w", buffering=1)  # line-buffered

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log(f"run_id: {run_id}")
    log(f"dataset: {args.dataset}")
    log(f"model_path: {args.model_path}")
    log(f"dataset_root: {args.dataset_root}")
    log(f"split: {args.split}")
    log(f"seed: {args.seed}")
    log(f"max_samples: {args.max_samples}")
    log(f"candidate_file: {args.candidate_file if not args.skip_open_scoring else '<skipped>'}")
    log(f"skip_open_scoring: {args.skip_open_scoring}")

    # === Load model =========================================================
    log("loading model...")
    t0 = time.perf_counter()
    loaded = load_llava_med_v1(args.model_path)
    load_secs = time.perf_counter() - t0
    log(f"model loaded in {load_secs:.1f}s "
        f"(conv_mode={loaded.conv_mode}, image_token_len={loaded.image_token_len}, "
        f"mm_use_im_start_end={loaded.mm_use_im_start_end})")

    # === Load dataset =======================================================
    log(f"loading dataset ({args.dataset}, split={args.split})...")
    loader_class = DATASET_REGISTRY[args.dataset]["loader"]
    dataset = loader_class(root=args.dataset_root, split=args.split)
    samples = list(dataset)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    log(f"loaded {len(samples)} samples "
        f"(closed: {sum(1 for s in samples if s.answer_type == 'closed')}, "
        f"open: {sum(1 for s in samples if s.answer_type == 'open')})")

    # === Load candidate set (if scoring open) ===============================
    if args.skip_open_scoring:
        # Build a stub candidate set so score_predictions doesn't crash.
        # Open questions will still run inference; we zero out their metrics
        # in the report below since the stub candidate set produces nonsense.
        candidate_set = {"0": ["__skip__"]}
        log("open scoring skipped; closed yes/no scoring still active")
    else:
        log(f"loading candidate set...")
        with open(args.candidate_file) as f:
            candidate_set = json.load(f)
        log(f"candidate set: {len(candidate_set['0'])} entries")

    # === Run inference ======================================================
    log("starting inference...")
    log(f"method: {method.name}")
    runner = Runner(loaded, method=method, seed=args.seed)
    t0 = time.perf_counter()
    predictions = runner.run(samples, progress=True)
    inference_secs = time.perf_counter() - t0
    avg_latency_ms = 1000 * inference_secs / len(predictions)
    log(f"inference done in {inference_secs:.1f}s "
        f"({avg_latency_ms:.0f}ms/sample average)")

    # === Score ==============================================================
    log("scoring...")
    report = score_predictions(samples, [p.to_dict() for p in predictions], candidate_set)
    log(f"closed_yes_no_accuracy:  {report.closed_yes_no_accuracy:.4f}  "
        f"(n={report.num_closed_yes_no}/{report.num_closed_total})")
    if args.skip_open_scoring:
        # Zero out the open metrics in the report since they were computed
        # against a stub candidate set; the closed counts are still real.
        report.open_appearance_accuracy = 0.0
        report.open_exact_match = 0.0
        report.open_f1 = 0.0
        report.open_recall = 0.0
        report.open_precision = 0.0
        report.open_bleu_score = 0.0
        report.open_bleu_score_1 = 0.0
        report.open_bleu_score_2 = 0.0
        report.open_bleu_score_3 = 0.0
        log(f"open metrics skipped (--skip-open-scoring); "
            f"{report.num_open} open samples ran inference but were not scored")
    else:
        log(f"open_appearance_acc:     {report.open_appearance_accuracy:.4f}  "
            f"(n={report.num_open})")
        log(f"open_exact_match:        {report.open_exact_match:.4f}")
        log(f"open_f1:                 {report.open_f1:.4f}")
        log(f"open_recall:             {report.open_recall:.4f}")
        log(f"open_precision:          {report.open_precision:.4f}")
        log(f"open_bleu_score:         {report.open_bleu_score:.4f}")
        log(f"open_bleu_score_1:       {report.open_bleu_score_1:.4f}")
        log(f"open_bleu_score_2:       {report.open_bleu_score_2:.4f}")
        log(f"open_bleu_score_3:       {report.open_bleu_score_3:.4f}")

    # === Write outputs ======================================================
    full_metrics = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config": {
            "dataset": args.dataset,
            "model_path": args.model_path,
            "dataset_root": args.dataset_root,
            "split": args.split,
            "seed": args.seed,
            "keep_ratio": args.keep_ratio if args.keep_ratio is not None else 1.0,
            "max_samples": args.max_samples,
            "candidate_file": args.candidate_file if not args.skip_open_scoring else None,
            "skip_open_scoring": args.skip_open_scoring,
        },
        "wall_time_seconds": {
            "model_load": load_secs,
            "inference": inference_secs,
            "avg_latency_ms_per_sample": avg_latency_ms,
        },
        "metrics": report.to_dict(),
    }
    with metrics_path.open("w") as f:
        json.dump(full_metrics, f, indent=2)
    log(f"wrote {metrics_path}")

    with preds_path.open("w") as f:
        for pred_rec, sample, qreport in zip(
            predictions, samples, report.per_question
        ):
            rec = {
                **pred_rec.to_dict(),
                "ground_truth": sample.answer,
                "answer_type": sample.answer_type,
                "scoring": {
                    k: v for k, v in qreport.items()
                    if k not in {"question_id", "answer_type", "gt", "pred"}
                },
            }
            f.write(json.dumps(rec) + "\n")
    log(f"wrote {preds_path}")

    log("done.")
    log_f.close()


if __name__ == "__main__":
    main()