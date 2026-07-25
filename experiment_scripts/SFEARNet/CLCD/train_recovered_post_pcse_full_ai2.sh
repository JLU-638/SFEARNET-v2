#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
AI2_ROOT="/fs1/private/user/fengruyue/lvguixin/SFEARNet_test"
STAMP="$(date +%F_%H-%M-%S)"
RUN_ROOT="${AI2_ROOT}/recovered_runs/post_pcse_20260627_full/run_${STAMP}"
WORKDIR="${RUN_ROOT}/workspace"
LOG_FILE="${AI2_ROOT}/nohup_logs/recovered_post_pcse_full_${STAMP}_gpu${GPU_ID}.log"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

mkdir -p "${RUN_ROOT}" "${AI2_ROOT}/nohup_logs"
rsync -a --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'logs' \
  --exclude 'results' \
  --exclude 'manuscript' \
  --exclude 'baseline_runs' \
  --exclude 'all_predictions' \
  --exclude 'recovered_runs' \
  --exclude 'data/pseudo_change_subsets' \
  "${AI2_ROOT}/" "${WORKDIR}/"

cd "${WORKDIR}"
cp recovered_code/post_pcse_20260627_guess/train.py train.py
cp recovered_code/post_pcse_20260627_guess/utils/stages.py utils/stages.py
cp recovered_code/post_pcse_20260627_guess/utils/args_utils.py utils/args_utils.py
cp recovered_code/post_pcse_20260627_guess/model/sfearnet/__init__.py model/sfearnet/__init__.py
cp recovered_code/post_pcse_20260627_guess/model/sfearnet/pyramid.py model/sfearnet/pyramid.py
cp recovered_code/post_pcse_20260627_guess/loss/PseudoChangesSuppression.py loss/PseudoChangesSuppression.py

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
nohup "${PYTHON_BIN}" -u train.py \
  --data_dir ./data/datasets/CLCD_256/ \
  --result_dir ./CLCD_result_recovered_post_pcse_full/ \
  --data CLCD \
  --model SFEARNet_recovered_post_pcse_full \
  --train_batchsize 8 \
  --val_batchsize 1 \
  --lr 0.0001 \
  --num_epochs 100 \
  --gpu_id 0 \
  --lr_decline ReduceLROnPlateau \
  --weight_decay 0.001 \
  --lamda 1.0 \
  --alpha 0.5 \
  --beta 0.1 \
  --gamma 0.4 \
  --diff_mode hybrid_cos \
  --seed 0 \
  --print_every_batches 100 \
  --num_workers 4 \
  --pin_memory true \
  --persistent_workers true \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "RUN_ROOT=${RUN_ROOT}"
echo "WORKDIR=${WORKDIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "PID=${PID}"
