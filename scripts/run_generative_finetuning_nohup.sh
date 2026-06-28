#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/generative_train_$(date +%Y%m%d_%H%M%S).log"

nohup python generative_finetuning.py \
  --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \
  --pretrain_ckpt checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt \
  --output_dir checkpoints_generative_qa \
  --decoder_variant hybrid \
  --max_input_len 256 \
  --max_target_len 48 \
  --train_batch_size 6 \
  --eval_batch_size 8 \
  --grad_accum 2 \
  --epochs 6 \
  --lr 3e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.06 \
  --max_grad_norm 1.0 \
  --label_smoothing 0.05 \
  --freeze_warmup_epochs 2 \
  --unfreeze_top_layers 4 \
  --num_workers 4 \
  --fp16 > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_DIR}/generative_train.pid"
echo "Started generative QA training."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "Watch: tail -f ${LOG_FILE}"
