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
GPU_ID="0"
SEED=0
# endregion

# region --- 数据集相关参数 ---
DATASET_TAG="LEVIR_CD"
DATASET_DIR="./data/datasets/LEVIR_CD_256/"
# endregion

# region --- 模型相关参数 ---
MODEL_NAME="SFEARNet"
# endregion

# region --- 训练与日志相关参数 ---
NUM_EPOCHS=100
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=1
PRINT_EVERY_BATCHES=100
RESULT_DIR="./LEVIR_CD_result_1/"
# endregion

# region --- 优化器与损失相关参数 ---
LR=1e-4
WEIGHT_DECAY=1e-3
LAMBDA_EDGE=1
LR_SCHEDULER="ReduceLROnPlateau"
# endregion

SAVE_DIR="./logs_of_shell/${MODEL_NAME}-${DATASET_TAG}_start-at-$(date +%Y%m%d_%H%M%S)"
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
    >"${SAVE_DIR}/training.log" 2>&1 &

echo "Experiment is running in background. Log saved at: ${SAVE_DIR}/training.log"