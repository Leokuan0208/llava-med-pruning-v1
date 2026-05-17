# LLaVA-Med Pruning Evaluation & Fine-Tuning Harness (v1.0)

Evaluation and fine-tuning infrastructure for **question-aware visual token
pruning** research on medical vision-language models, using **LLaVA-Med v1.0
(Vicuna-7B)** as the baseline.

Decoupled from the LLaVA-Med codebase; the harness imports `llava` as a
library at runtime. This repository is the *measurement and training
infrastructure* for the project — it runs a model over medical VQA benchmarks
(optionally with a pruning method attached), reports accuracy and latency, and
can full fine-tune the v1.0 stage-2 checkpoint on per-dataset training data.
The pruning methods themselves are the research contribution and are added on
top of this harness.

This is the v1.0 branch of the project; the v1.5 (Mistral-7B) variant is
maintained separately. Both share the same overall harness shape (datasets,
methods, runner, metrics) with version-specific adaptations to scoring and
inference recipes.

## Status

Evaluation harness is functional end-to-end for VQA-RAD and PathVQA. SLAKE
loader pending (yesterday's published delta was an empty repo on HuggingFace;
SLAKE evaluation depends on either obtaining a working delta or fine-tuning
from stage-2 directly).

- [x] Dataset / method base interfaces
- [x] Metrics module (v1.0 scoring: closed yes/no accuracy, open-question
      appearance accuracy + BLEU + F1, byte-for-byte port of v1.0's reference)
- [x] Model loader wrapper (LLaVA-Med v1.0)
- [x] Runner (stochastic decoding at T=0.7, KeywordsStoppingCriteria,
      conv-template-aware post-truncation)
- [x] CLI entry point with multi-dataset registry
- [x] VQA-RAD loader (with real `answer_type` labels joined from the original
      VQA-RAD distribution)
- [x] PathVQA loader (yes/no heuristic for `answer_type`; original-distribution
      join is a future improvement)
- [x] Candidate-set builders for v1.0 open-question scoring (VQA-RAD: 402
      entries; PathVQA: 3,223 entries)
- [x] Full fine-tuning pipeline (DeepSpeed Zero-2 with CPU optimizer offload)
- [x] Multi-checkpoint test-set eval and best-by-accuracy selection
- [ ] SLAKE loader
- [ ] Pruning methods (the research contribution — pending baseline validation)

## Key findings so far

LLaVA-Med v1.0's published per-dataset fine-tuned deltas
(`microsoft/llava-med-7b-{vqarad,pathvqa}-delta`) do not reproduce the paper's
reported Table 4 numbers when merged with LLaMA-7B and evaluated with the
reference inference recipe. Specifically:

- VQA-RAD-merged: 0.21 closed yes/no accuracy (paper: 0.836)
- Stage-2 zero-shot on VQA-RAD: 0.58 closed (paper's stage-2 row: ~0.50)

The stage-2 result is within stochastic-decoding noise of the paper, validating
the harness, base LLaMA-7B, merge process, and inference recipe. The
discrepancy is upstream: Microsoft's published per-dataset deltas appear
corrupted, and the public repo contains no working stage-3 fine-tuning recipe
(`scripts/chunyl/finetune_on_benchmarks/fine_tuning_*_7B.sh` is a stage-1
projector-only ablation despite the misleading name).

Our own 5-epoch full fine-tune of stage-2 → VQA-RAD train is in progress;
results will be added once available.

## Layout
llava-med-pruning-v1/
├── eval/
│   ├── runner.py                       # Inference runner (v1.0 recipe)
│   ├── metrics.py                      # v1.0 scoring (port of reference)
│   ├── model_loader.py                 # Wraps v1.0 model loading
│   ├── datasets/
│   │   ├── base.py                     # MedVQADataset interface + VQASample
│   │   ├── vqa_rad.py                  # VQA-RAD loader (parquet + answer_type join)
│   │   └── path_vqa.py                 # PathVQA loader (sharded parquet)
│   ├── v1_assets/
│   │   ├── vqa_rad_train_open_answers.json   # 402 candidates
│   │   └── path_vqa_train_open_answers.json  # 3,223 candidates
│   ├── build_vqa_rad_train_open_answers.py
│   └── build_path_vqa_train_open_answers.py
├── scripts/
│   ├── run_E0_v1.py                    # Multi-dataset eval entrypoint
│   ├── build_vqarad_train_v1_format.py # Train data → v1.0 conversation JSON
│   ├── finetune_smoke.sh               # 30-step training smoke test
│   ├── finetune_vqarad_full.sh         # Full FT (5 epochs, paper-faithful recipe)
│   ├── eval_topk_checkpoints.py        # Eval all FT checkpoints, pick best
│   └── ds_config_zero2_bf16.json       # DeepSpeed Zero-2 + CPU offload config
└── results/                            # Evaluation outputs

## Output format

Each evaluation run writes three files to `results/`:

1. `<run_id>_metrics.json` — aggregate metrics plus the fully resolved config
   (every CLI argument is saved, so a run is reproducible from this file alone).
2. `<run_id>_predictions.jsonl` — one JSON object per evaluated question, for
   error analysis and qualitative inspection. (Gitignored; regenerable.)
3. `<run_id>.log` — timestamped milestones from the run.

`<run_id>` defaults to `E0_v1.0_{dataset}_{split}_{timestamp}` or can be
overridden via `--run-id`.

## Usage

### Evaluation

```bash
PYTHONPATH=.:$HOME/LLaVA-Med-v1.0 python scripts/run_E0_v1.py \
    --dataset vqa_rad \
    --model-path /data/dan/weights/llava-med-7b-stage2-merged \
    --run-id stage2_baseline_vqarad
```

Multi-dataset support: pass `--dataset path_vqa` and the registry picks the
right loader, candidate file, and default data root automatically. Auto-falls
back to closed-only scoring if the per-dataset candidate file isn't built yet.

### Full fine-tuning

```bash
# One-time: build the v1.0-format training JSON
PYTHONPATH=. python scripts/build_vqarad_train_v1_format.py

# Optional but recommended: smoke test (5 min, 30 steps)
./scripts/finetune_smoke.sh

# Real run (background, ~6 hours)
nohup ./scripts/finetune_vqarad_full.sh > finetune_vqarad_full.nohup.log 2>&1 &
```

The full FT script auto-runs `eval_topk_checkpoints.py` at the end, so the
result `topk_summary.json` appears in the output directory automatically.

## Notes on dataset loaders

**VQA-RAD.** Loaded from the HuggingFace parquet mirror
(`flaviagiammarino/vqa-rad`), which does not carry the original dataset's
`answer_type` field. Real closed/open labels are restored by joining against
the original VQA-RAD distribution (`VQA_RAD Dataset Public.json`) on a
normalized `(question, answer)` key. As of the current dataset copies, 450/451
test samples join successfully; 1 falls back to a heuristic label (documented
in `vqa_rad.py`).

**PathVQA.** Loaded from the sharded parquet mirror
(`flaviagiammarino/path-vqa`). The mirror also lacks an `answer_type` column,
and the original distribution join hasn't been built yet — we use the yes/no
heuristic (`yes` or `no` → closed, otherwise open). PathVQA's closed questions
are essentially all yes/no by design, so the mislabel rate is expected to be
very low.

Parquet-embedded images are materialized to disk once, on first load, under
`<dataset-root>/extracted_images/<split>/`. Subsequent runs reuse the cached
images.

## Dependencies

Requires the NVIDIA NGC PyTorch container `nvcr.io/nvidia/pytorch:23.10-py3`
plus LLaVA-Med v1.0 (https://github.com/microsoft/LLaVA-Med) checked out at
the v1.0 commit, alongside this harness. Key in-container additions on top of
NGC's defaults:

- `transformers==4.28.0.dev0` (v1.0's pinned commit, NOT a release version)
- `tokenizers==0.12.1`
- `deepspeed==0.9.5`
- `pydantic<2` (required by DS 0.9.5's API)
- `bitsandbytes==0.41.0`
- `open_clip_torch==2.23.0`, `timm==0.9.12`, `ftfy`, `regex`, `nltk`,
  `py-cpuinfo`, `psutil`, `hjson`, `ninja`

See companion infrastructure repo (TBD) for the full Dockerfile and rebuild
instructions.

## Reproducibility caveats

Stochastic decoding (`temperature=0.7`) means individual runs vary by ~1-2
points; the smoke-test loss and final accuracy will differ slightly between
seeds. For the baseline reproduction findings above we used `--seed 42`.