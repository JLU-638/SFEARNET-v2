#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
AI2_ROOT="/fs1/private/user/fengruyue/lvguixin/SFEARNet_test"
SMOKE_ROOT="${AI2_ROOT}/recovered_runs/post_pcse_20260627_smoke"
STAMP="$(date +%F_%H-%M-%S)"
RUN_ROOT="${SMOKE_ROOT}/run_${STAMP}"
LOG_FILE="${AI2_ROOT}/nohup_logs/recovered_post_pcse_smoke_${STAMP}_gpu${GPU_ID}.log"

mkdir -p "${RUN_ROOT}"
mkdir -p "${AI2_ROOT}/nohup_logs"

rsync -a --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'logs' \
  --exclude 'results' \
  --exclude 'manuscript' \
  --exclude 'baseline_runs' \
  --exclude 'all_predictions' \
  --exclude 'data/pseudo_change_subsets' \
  "${AI2_ROOT}/" "${RUN_ROOT}/workspace/"

cd "${RUN_ROOT}/workspace"

cp recovered_code/post_pcse_20260627_guess/train.py train.py
cp recovered_code/post_pcse_20260627_guess/utils/stages.py utils/stages.py
cp recovered_code/post_pcse_20260627_guess/utils/args_utils.py utils/args_utils.py
cp recovered_code/post_pcse_20260627_guess/model/sfearnet/__init__.py model/sfearnet/__init__.py
cp recovered_code/post_pcse_20260627_guess/model/sfearnet/pyramid.py model/sfearnet/pyramid.py
cp recovered_code/post_pcse_20260627_guess/loss/PseudoChangesSuppression.py loss/PseudoChangesSuppression.py

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
nohup /usr/bin/python3 -u train.py \
  --data_dir ./data/datasets/CLCD_256/ \
  --data CLCD \
  --model SFEARNet_recovered_post_pcse_smoke \
  --train_batchsize 8 \
  --val_batchsize 1 \
  --lr 0.0001 \
  --num_epochs 12 \
  --gpu_id 0 \
  --weight_decay 0.001 \
  --lamda 1.0 \
  --alpha 0.5 \
  --beta 0.1 \
  --gamma 0.4 \
  --diff_mode hybrid_cos \
  --seed 0 \
  --num_workers 4 \
  --pin_memory true \
  --persistent_workers true \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "RUN_ROOT=${RUN_ROOT}"
echo "LOG_FILE=${LOG_FILE}"
echo "PID=${PID}"
