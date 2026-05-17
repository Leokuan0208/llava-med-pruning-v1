#!/bin/bash
# 5-epoch full fine-tuning of LLaVA-Med v1.0 stage-2 on VQA-RAD train.
#
# Starts from the verified-working stage-2 merged checkpoint (NOT the broken
# vqa-rad-merged from HuggingFace). Goal: test whether ~5 epochs of full FT
# from stage-2 gives a meaningful accuracy bump on VQA-RAD test, indirectly
# answering whether the paper's published delta is genuinely broken vs whether
# our pipeline is at fault.
#
# Memory configuration: DeepSpeed Zero-2 with optimizer state offloaded to
# CPU. Keeps the 28 GB fp32 AdamW master + 28 GB momentum + 28 GB variance
# on CPU RAM (we have 250 GB), frees GPU for params + grads + activations.
# Observed GPU usage: ~28-50 GB depending on batch.
#
# Per-step time: ~20 sec, limited by PCIe transfer to/from CPU optimizer
# state. Cannot be reduced meaningfully without changing optimizer or
# accepting OOM risk.
#
# Steps per epoch: 1,793 samples / 8 batch = ~225 steps.
# Total: 225 * 5 = 1,125 steps, ~6 hours wall time.
#
# Output: /data/dan/weights/llava-med-7b-vqarad-ft-15ep/
#   3 most recent checkpoints retained (save_total_limit=3). After training,
#   eval_topk_checkpoints.py runs E0_v1.0 on each surviving checkpoint
#   against VQA-RAD test, picks best by closed accuracy.

set -euo pipefail

LLAVA_MED_ROOT="${HOME}/LLaVA-Med-v1.0"
STAGE2_MODEL="/data/dan/weights/llava-med-7b-stage2-merged"
TRAIN_JSON="/data/dan/dataset/vqa_rad/vqa_rad_train_v1_format.json"
IMAGE_FOLDER="/data/dan/dataset/vqa_rad/extracted_images/train"
OUTPUT_DIR="/data/dan/weights/llava-med-7b-vqarad-ft-15ep"
DS_CONFIG="${HOME}/llava-med-pruning-v1/scripts/ds_config_zero2_bf16.json"

# Pre-flight: ensure inputs exist before launching 6 hours of training.
test -d "${STAGE2_MODEL}"   || { echo "MISSING: ${STAGE2_MODEL}"; exit 1; }
test -f "${TRAIN_JSON}"     || { echo "MISSING: ${TRAIN_JSON}"; exit 1; }
test -d "${IMAGE_FOLDER}"   || { echo "MISSING: ${IMAGE_FOLDER}"; exit 1; }
test -f "${DS_CONFIG}"      || { echo "MISSING: ${DS_CONFIG}"; exit 1; }

mkdir -p "${OUTPUT_DIR}"

cd "${LLAVA_MED_ROOT}"

# Export PYTHONPATH so the subprocess spawned by deepspeed launcher can find
# the `llava` package. cd-ing into the dir isn't enough -- Python only auto-
# adds CWD to sys.path for direct script invocation, not for the launcher's
# subprocess.
export PYTHONPATH="${LLAVA_MED_ROOT}:${PYTHONPATH:-}"

# Use `python -m deepspeed.launcher.runner` instead of the bare `deepspeed`
# CLI, because the latter requires the deepspeed CLI binary to be on PATH
# (it isn't in our container; pip installed user-mode).
python -m deepspeed.launcher.runner --num_gpus=1 --master_port=25001 \
    llava/train/train_mem.py \
    --deepspeed "${DS_CONFIG}" \
    --model_name_or_path "${STAGE2_MODEL}" \
    --data_path "${TRAIN_JSON}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --tf32 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to "none" \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

echo
echo "=== Training complete ==="
echo "Surviving checkpoints:"
ls -la "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null || echo "WARNING: no checkpoints saved"

# === Auto-run eval on all surviving checkpoints ===========================
echo
echo "=== Running test-set eval on all surviving checkpoints ==="
cd "${HOME}/llava-med-pruning-v1"
export PYTHONPATH=".:${HOME}/LLaVA-Med-v1.0:${PYTHONPATH:-}"
python scripts/eval_topk_checkpoints.py \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/eval_topk.log"

echo
echo "=== Pipeline complete ==="
echo "Training log:  ${OUTPUT_DIR}/training.log"
echo "Eval log:      ${OUTPUT_DIR}/eval_topk.log"
echo "Top-k summary: ${OUTPUT_DIR}/topk_summary.json"