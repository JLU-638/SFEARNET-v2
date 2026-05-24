#!/bin/bash
set -e
set -u
set -o pipefail

# --- 强力清理代理，防止外部网络依赖报错 ---
unset http_proxy
unset https_proxy
unset ALL_PROXY

# 进入项目根目录
cd "$(dirname "$0")/../../.."

# region --- 基础环境配置 ---
GPU_ID=0
SEED=0
# endregion

# region --- 数据集相关参数 ---
DATASET_TAG="CLCD"
DATASET_DIR="./data/datasets/CLCD_256/"
# endregion

# region --- 模型相关参数 ---
MODEL_NAME="SFEARNet"
# endregion

# region --- 训练与日志相关参数 ---
NUM_EPOCHS=100
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=1
PRINT_EVERY_BATCHES=100
NUM_WORKERS=4
PIN_MEMORY=true
PERSISTENT_WORKERS=true
RESULT_DIR="./CLCD_result_1/"
# endregion

# region --- 优化器与损失相关参数 ---
LR=0.0001
WEIGHT_DECAY=0.001
LAMBDA_EDGE=1
LR_SCHEDULER="ReduceLROnPlateau"
# endregion

RUN_TIME=$(date +%Y-%m-%d_%H-%M-%S)
EXP_LEAF="lr-${LR}_bs-${TRAIN_BATCH_SIZE}_wd-${WEIGHT_DECAY}_lam-${LAMBDA_EDGE}_ep-${NUM_EPOCHS}_seed-${SEED}"
SAVE_DIR="./logs_of_shell/${DATASET_TAG}/${MODEL_NAME}/${EXP_LEAF}/run_${RUN_TIME}"
mkdir -p "$SAVE_DIR"

echo "Starting Experiment: model=${MODEL_NAME}, data=${DATASET_TAG}"
echo "Shell log dir: ${SAVE_DIR}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# 使用 nohup 配合 &，退出终端后训练继续，日志写入文件
nohup python train.py \
    --data_dir "$DATASET_DIR" \
    --result_dir "$RESULT_DIR" \
    --data "$DATASET_TAG" \
    --train_batchsize "$TRAIN_BATCH_SIZE" \
    --val_batchsize "$VAL_BATCH_SIZE" \
    --lr "$LR" \
    --num_epochs "$NUM_EPOCHS" \
    --gpu_id "$GPU_ID" \
    --model "$MODEL_NAME" \
    --lr_decline "$LR_SCHEDULER" \
    --weight_decay "$WEIGHT_DECAY" \
    --lamda "$LAMBDA_EDGE" \
    --seed "$SEED" \
    --print_every_batches "$PRINT_EVERY_BATCHES" \
    --num_workers "$NUM_WORKERS" \
    --pin_memory "$PIN_MEMORY" \
    --persistent_workers "$PERSISTENT_WORKERS" \
    >"${SAVE_DIR}/training.log" 2>&1 &
# 可选：添加 --max_test_batches 非零值（例如 200）可只跑每个 epoch 的前 N 个 batch，便于快速联调。

echo "Experiment is running in background. Log saved at: ${SAVE_DIR}/training.log"