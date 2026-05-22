"""run_eval_v2: MCQ-based closed scoring + token-level open scoring.

Replaces scripts/run_E0_v1.py for the actual research evaluation. Key
differences:

1. Closed yes/no questions are reformulated as 2-option MCQ (A/B with
   randomized letter assignment per question_id) before inference.
2. Closed scoring is letter-extraction-based, no substring matching.
3. Non-yes/no closed questions (~7.7% of VQA-RAD, ~14.7% of SLAKE, 0% of
   PathVQA) run inference with their original question but are bucketed
   as `unscorable_by_mcq` rather than scored.
4. The `appearance_accuracy` open metric and its candidate-file dependency
   are removed. The remaining open metrics (exact_match, F1, precision,
   recall, BLEU 1/2/3/4) are reused unchanged.

Expected outputs:
  results/<run_id>_metrics.json     -- aggregate metrics (V2 schema)
  results/<run_id>_predictions.jsonl -- per-question records including
      MCQ-formatted prompts and extracted letters for closed yes/no samples
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import replace as dc_replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.model_loader import load_llava_med_v1
from eval.runner import Runner
from eval.mcq import format_yesno_as_mcq
from eval.scoring_v2 import score_predictions_v2
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

DATASET_REGISTRY = {
    "vqa_rad":  {"loader": VQARadDataset,  "data_root": "/data/dan/dataset/vqa_rad"},
    "path_vqa": {"loader": PathVQADataset, "data_root": "/data/dan/dataset/path_vqa"},
    "slake":    {"loader": SlakeDataset,   "data_root": "/data/dan/dataset/slake"},
}

DEFAULT_MODEL_PATH = "/data/dan/weights/llava-med-baron-gg-stage2"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args():
    p = argparse.ArgumentParser(description="run_eval_v2: MCQ-based medical VQA eval")
    p.add_argument("--dataset", default="vqa_rad", choices=sorted(DATASET_REGISTRY.keys()))
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default=None)
    p.add_argument("--method", default="baseline", choices=sorted(METHOD_REGISTRY.keys()))
    p.add_argument("--keep-ratio", type=float, default=1.0)
    p.add_argument("--mcq-seed", type=int, default=12345,
                   help="Seed for MCQ letter randomization (A/B assignment). "
                        "Stable per (question_id, mcq_seed) so re-runs produce "
                        "the same MCQ formatting.")
    args = p.parse_args()
    if args.dataset_root is None:
        args.dataset_root = DATASET_REGISTRY[args.dataset]["data_root"]
    return args


def _build_mcq_samples(samples, mcq_seed):
    """Reformulate yes-no closed samples as MCQ; pass everything else through.

    Returns:
        (mcq_samples, mcq_metadata)
        mcq_samples: list of VQASample, identical to input except yes-no closed
            have their `question` field replaced with the MCQ prompt.
        mcq_metadata: dict question_id -> {correct_letter, options,
            formatted_question} for every yes-no closed sample.
    """
    mcq_samples = []
    mcq_metadata = {}
    for s in samples:
        at = str(s.answer_type).strip().lower()
        gt_norm = str(s.answer).strip().lower()
        is_yesno = (at == "closed" and gt_norm in ("yes", "no"))

        if not is_yesno:
            mcq_samples.append(s)
            continue

        # Per-sample deterministic seed: combine global mcq_seed with the
        # hash of question_id so A/B assignment is stable across runs but
        # different per question.
        sample_seed = hash((mcq_seed, s.question_id)) & 0xFFFFFFFF
        fmt = format_yesno_as_mcq(s.question, s.answer, sample_seed=sample_seed)

        new_sample = dc_replace(s, question=fmt.formatted_question)
        mcq_samples.append(new_sample)
        mcq_metadata[s.question_id] = {
            "correct_letter": fmt.correct_letter,
            "options": list(fmt.options),
            "formatted_question": fmt.formatted_question,
        }
    return mcq_samples, mcq_metadata


def main():
    args = parse_args()
    method = METHOD_REGISTRY[args.method](args.keep_ratio)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"E1_v2_{args.dataset}_{method.name}_{args.split}_{timestamp}"

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"{run_id}_metrics.json"
    preds_path = results_dir / f"{run_id}_predictions.jsonl"
    log_path = results_dir / f"{run_id}.log"

    log_f = log_path.open("w", buffering=1)
    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log(f"run_id: {run_id}")
    log(f"dataset: {args.dataset}  split: {args.split}")
    log(f"model_path: {args.model_path}")
    log(f"method: {method.name}  keep_ratio: {args.keep_ratio}")
    log(f"seed: {args.seed}  mcq_seed: {args.mcq_seed}")

    # === Load model =========================================================
    log("loading model...")
    t0 = time.perf_counter()
    loaded = load_llava_med_v1(args.model_path)
    load_secs = time.perf_counter() - t0
    log(f"model loaded in {load_secs:.1f}s")

    # === Load dataset =======================================================
    log(f"loading dataset...")
    loader_class = DATASET_REGISTRY[args.dataset]["loader"]
    dataset = loader_class(root=args.dataset_root, split=args.split)
    samples = list(dataset)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    n_closed = sum(1 for s in samples if s.answer_type == "closed")
    n_open = sum(1 for s in samples if s.answer_type == "open")
    n_yesno = sum(
        1 for s in samples
        if s.answer_type == "closed"
        and str(s.answer).strip().lower() in ("yes", "no")
    )
    n_unscorable = n_closed - n_yesno
    log(f"loaded {len(samples)} samples "
        f"(closed: {n_closed} [yes/no: {n_yesno}, unscorable: {n_unscorable}], "
        f"open: {n_open})")

    # === Build MCQ samples ==================================================
    log("building MCQ-formatted prompts for yes-no closed samples...")
    mcq_samples, mcq_metadata = _build_mcq_samples(samples, mcq_seed=args.mcq_seed)
    log(f"MCQ metadata built for {len(mcq_metadata)} yes-no closed samples")

    # === Run inference ======================================================
    log("starting inference...")
    runner = Runner(loaded, method=method, seed=args.seed)
    t0 = time.perf_counter()
    predictions = runner.run(mcq_samples, progress=True)
    inference_secs = time.perf_counter() - t0
    avg_latency_ms = 1000 * inference_secs / max(1, len(predictions))
    log(f"inference done in {inference_secs:.1f}s ({avg_latency_ms:.0f}ms/sample)")

    # === Score ==============================================================
    log("scoring (V2: MCQ + token-level)...")
    # Note: ORIGINAL samples passed to scorer (we need the GT answer; the
    # MCQ-mutated question is irrelevant to scoring).
    report = score_predictions_v2(
        samples, [p.to_dict() for p in predictions], mcq_metadata
    )

    log(f"--- Closed (MCQ) ---")
    log(f"  mcq_accuracy:          {report.mcq_accuracy:.4f}  (extracted-only)")
    log(f"  mcq_strict_accuracy:   {report.mcq_strict_accuracy:.4f}  (failures=wrong)")
    log(f"  mcq_extraction_rate:   {report.mcq_extraction_rate:.4f}  "
        f"({report.num_closed_yesno - report.num_extraction_failed}/{report.num_closed_yesno})")
    log(f"  num_closed_yesno:      {report.num_closed_yesno}")
    log(f"  num_closed_unscorable: {report.num_closed_unscorable}")
    log(f"  num_extraction_failed: {report.num_extraction_failed}")
    log(f"--- Open (token-level) ---")
    log(f"  open_exact_match:      {report.open_exact_match:.4f}")
    log(f"  open_f1:               {report.open_f1:.4f}")
    log(f"  open_precision:        {report.open_precision:.4f}")
    log(f"  open_recall:           {report.open_recall:.4f}")
    log(f"  open_bleu (cum-4):     {report.open_bleu_score:.4f}")
    log(f"  num_open:              {report.num_open}")

    # === Write outputs ======================================================
    full_metrics = {
        "run_id": run_id,
        "timestamp": timestamp,
        "scorer_version": "v2",
        "config": {
            "dataset": args.dataset,
            "model_path": args.model_path,
            "dataset_root": args.dataset_root,
            "split": args.split,
            "seed": args.seed,
            "mcq_seed": args.mcq_seed,
            "method": args.method,
            "keep_ratio": args.keep_ratio,
            "max_samples": args.max_samples,
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
        for pred_rec, sample, qreport in zip(predictions, samples, report.per_question):
            rec = {
                **pred_rec.to_dict(),
                "ground_truth": sample.answer,
                "answer_type": sample.answer_type,
                "original_question": sample.question,
                "scoring": {
                    k: v for k, v in qreport.items()
                    if k not in {"question_id", "answer_type", "gt", "pred"}
                },
            }
            if sample.question_id in mcq_metadata:
                rec["mcq_metadata"] = mcq_metadata[sample.question_id]
            f.write(json.dumps(rec) + "\n")
    log(f"wrote {preds_path}")
    log("done.")
    log_f.close()


if __name__ == "__main__":
    main()
