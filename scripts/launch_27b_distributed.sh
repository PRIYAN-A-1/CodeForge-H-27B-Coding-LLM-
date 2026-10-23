#!/usr/bin/env bash
# ==============================================================================
# CodeForge AI - Production 27B Distributed Training Launch Script
# Target Architecture: Multi-GPU Server (4x / 8x A100 80GB or H100 80GB)
# Framework: PyTorch Distributed (torchrun) + DeepSpeed ZeRO-3
# ==============================================================================

set -euo pipefail

NUM_GPUS=${1:-8}
CONFIG_FILE=${2:-"configs/27b_sft_qlora.yaml"}
DEEPSPEED_CONFIG="configs/27b_deepspeed_zero3.json"
OUTPUT_DIR="checkpoints/27b_run_$(date +%Y%m%d_%H%M%S)"

echo "======================================================================="
echo "  CODEFORGE AI - 27B DISTRIBUTED TRAINING LAUNCHER"
echo "======================================================================="
echo "GPUs Allocated:       ${NUM_GPUS}"
echo "Configuration:        ${CONFIG_FILE}"
echo "DeepSpeed Config:     ${DEEPSPEED_CONFIG}"
echo "Output Checkpoints:   ${OUTPUT_DIR}"
echo "======================================================================="

# Verify GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. This script requires a Linux host with NVIDIA accelerators."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export PYTHONUNBUFFERED=1

# Launch distributed training via torchrun
torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29500 \
    scripts/train.py \
    --model 27b \
    --config "${CONFIG_FILE}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "logs/train_27b_${NUM_GPUS}gpu.log"

echo "[SUCCESS] Distributed training job finished. Checkpoints saved in ${OUTPUT_DIR}."
