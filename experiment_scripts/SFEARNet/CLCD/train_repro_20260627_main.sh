#!/bin/bash
set -e
set -u
set -o pipefail

unset http_proxy
unset https_proxy
unset ALL_PROXY

cd "$(dirname "$0")/../../.."

GPU_ID="${GPU_ID:-0}"
SEED=0

DATASET_TAG="CLCD"
DATASET_DIR="./data/datasets/CLCD_256/"

MODEL_NAME="SFEARNet_modified"

NUM_EPOCHS=100
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=1
PRINT_EVERY_BATCHES=100
NUM_WORKERS=4
PIN_MEMORY=true
PERSISTENT_WORKERS=true

LR=0.0001
WEIGHT_DECAY=0.001
LAMBDA_EDGE=1.0
ALPHA=0.5
BETA=0.1
GAMMA=0.4
DIFF_MODE="hybrid_cos"
PSEUDO_MARGIN=3.0

USE_PSEUDO_SUPPRESSION=true
USE_EDGE_SUPERVISION=true
USE_BOUNDED_OBJECTIVE=true

RESULT_DIR="./CLCD_result_modified/"

RUN_TIME=$(date +%Y-%m-%d_%H-%M-%S)
EXP_LEAF="lr-${LR}_bs-${TRAIN_BATCH_SIZE}_wd-${WEIGHT_DECAY}_lam-${LAMBDA_EDGE}_ep-${NUM_EPOCHS}_seed-${SEED}"
SAVE_DIR="./logs_of_shell/${DATASET_TAG}/${MODEL_NAME}/${EXP_LEAF}/run_${RUN_TIME}"
mkdir -p "$SAVE_DIR"

echo "Starting repro run for 2026-06-27 CLCD main experiment"
echo "Shell log dir: ${SAVE_DIR}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

nohup python -u train.py \
  --data_dir "$DATASET_DIR" \
  --result_dir "$RESULT_DIR" \
  --data "$DATASET_TAG" \
  --train_batchsize "$TRAIN_BATCH_SIZE" \
  --val_batchsize "$VAL_BATCH_SIZE" \
  --lr "$LR" \
  --num_epochs "$NUM_EPOCHS" \
  --gpu_id "$GPU_ID" \
  --model "$MODEL_NAME" \
  --lr_decline "ReduceLROnPlateau" \
  --weight_decay "$WEIGHT_DECAY" \
  --lamda "$LAMBDA_EDGE" \
  --alpha "$ALPHA" \
  --beta "$BETA" \
  --gamma "$GAMMA" \
  --diff_mode "$DIFF_MODE" \
  --use_pseudo_suppression "$USE_PSEUDO_SUPPRESSION" \
  --use_edge_supervision "$USE_EDGE_SUPERVISION" \
  --use_bounded_objective "$USE_BOUNDED_OBJECTIVE" \
  --pseudo_margin "$PSEUDO_MARGIN" \
  --seed "$SEED" \
  --print_every_batches "$PRINT_EVERY_BATCHES" \
  --num_workers "$NUM_WORKERS" \
  --pin_memory "$PIN_MEMORY" \
  --persistent_workers "$PERSISTENT_WORKERS" \
  >"${SAVE_DIR}/training.log" 2>&1 &

echo "Repro experiment is running in background."
echo "Log saved at: ${SAVE_DIR}/training.log"
