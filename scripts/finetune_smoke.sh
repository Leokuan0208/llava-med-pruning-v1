#!/bin/bash
# Finetune smoke test: 30 steps on VQA-RAD train, single epoch budget.
# Verifies: model loads, dataloader works, forward+backward+step works,
# GPU memory fits, and a checkpoint saves to disk. If this fails, the
# overnight full FT will fail in the same way -- fix here before sleeping.
#
# Expected wall time: ~3-5 minutes.
# Output dir: /data/dan/weights/llava-med-7b-vqarad-ft-smoke/

set -euo pipefail

LLAVA_MED_ROOT="${HOME}/LLaVA-Med-v1.0"
STAGE2_MODEL="/data/dan/weights/llava-med-7b-stage2-merged"
TRAIN_JSON="/data/dan/dataset/vqa_rad/vqa_rad_train_v1_format.json"
IMAGE_FOLDER="/data/dan/dataset/vqa_rad/extracted_images/train"
OUTPUT_DIR="/data/dan/weights/llava-med-7b-vqarad-ft-smoke"

# Pre-flight: ensure inputs exist before launching anything heavy.
test -d "${STAGE2_MODEL}"   || { echo "MISSING: ${STAGE2_MODEL}"; exit 1; }
test -f "${TRAIN_JSON}"     || { echo "MISSING: ${TRAIN_JSON} -- run build_vqarad_train_v1_format.py first"; exit 1; }
test -d "${IMAGE_FOLDER}"   || { echo "MISSING: ${IMAGE_FOLDER}"; exit 1; }

# Clean any prior smoke output so we test a fresh checkpoint save.
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

cd "${LLAVA_MED_ROOT}"

# Export PYTHONPATH so subprocess spawned by torchrun can find the `llava`
# package. cd-ing into the dir isn't enough -- Python only auto-adds the
# CWD to sys.path for direct script invocation, not for torchrun's
# elastic-launcher subprocess.
export PYTHONPATH="${LLAVA_MED_ROOT}:${PYTHONPATH:-}"

DS_CONFIG="${HOME}/llava-med-pruning-v1/scripts/ds_config_zero2_bf16.json"
test -f "${DS_CONFIG}" || { echo "MISSING: ${DS_CONFIG}"; exit 1; }

# Notes on what changed from the bnb-8bit attempt:
#   - Switched to deepspeed launcher instead of torchrun. DS adds the
#     necessary distributed init + bf16 optimizer path.
#   - --deepspeed ds_config_zero2_bf16.json: enables bf16 master weights
#     (28 GB savings vs fp32 master). This is the actual fix for the
#     OOM that bnb 8-bit couldn't solve alone.
#   - Dropped --optim adamw_bnb_8bit: DS provides its own AdamW now.
#     Keeping bnb installed but not used by the trainer.
#   - model_max_length back to 1024 since we have headroom now.
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
    --max_steps 30 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to "none" \
    2>&1 | tee "${OUTPUT_DIR}/smoke.log"

echo
echo "=== Smoke test complete ==="
ls -la "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null || echo "WARNING: no checkpoints saved"