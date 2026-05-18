#!/usr/bin/env bash
set -euo pipefail

# Batch inference and evaluation for the AGRI satellite tokenizer.
# Usage:
#   bash run_batch_evaluate_agri_tokenizer_region.sh \
#       --data_pattern 'datasets/AGRI_processed/preprocessed_video/*_agri_video_*.npz' \
#       --checkpoint checkpoints/satellite_tokenizer/agri/<run>/checkpoints/iter_000240000_ema.jit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

DATA_PATTERN="datasets/AGRI_processed/preprocessed_video/*_agri_video_*.npz"
CHECKPOINT=""
CHECKPOINT_ENC=""
CHECKPOINT_DEC=""
OUTPUT_DIR="outputs/agri_tokenizer_region_eval"
REGION=""
MAX_SAMPLES=""
CUDA_VISIBLE_DEVICES_DEFAULT="0"
DEVICE="cuda"
DTYPE="bfloat16"
USE_MASK_CHANNEL="true"
MASK_METRICS="true"
EVALUATE_MASK_CHANNEL="false"
SAVE_RECONSTRUCTIONS="false"
DATA_KEY="agri_data"
MASK_KEY="observation_mask"
PYTHON_BIN="python"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cosmos-predict1}"

print_help() {
    cat <<'EOF'
AGRI Tokenizer 区域批量推理与评估脚本

必需参数:
  --checkpoint PATH             完整 autoencoder JIT，例如 *_ema.jit
    或同时提供:
  --checkpoint_enc PATH         encoder JIT，例如 *_enc.jit
  --checkpoint_dec PATH         decoder JIT，例如 *_dec.jit

常用参数:
  --data_pattern PATTERN        AGRI NPZ glob 或 @filelist.txt
  --output_dir DIR              评估结果输出目录
  --region ROW0:ROW1,COL0:COL1  可选像素区域；不填表示全图
  --max_samples INT             最多评估样本数
  --cuda_visible_devices STR    可见 GPU ID，默认 0
  --device STR                  torch device，默认 cuda
  --dtype STR                   float32|float16|bfloat16，默认 bfloat16
  --use_mask_channel BOOL       输入是否使用第 10 个 mask 通道，默认 true
  --mask_metrics BOOL           指标是否只统计 observation_mask=1 区域，默认 true
  --evaluate_mask_channel BOOL  是否把 mask 通道也纳入指标，默认 false
  --save_reconstructions BOOL   是否保存重建 NPZ，默认 false
  --data_key KEY                NPZ 数据键，默认 agri_data
  --mask_key KEY                NPZ mask 键，默认 observation_mask
  --python_bin PATH             Python 可执行文件，默认 python
  --project_root PATH           项目根目录，默认脚本所在目录

示例:
  bash run_batch_evaluate_agri_tokenizer_region.sh \
      --data_pattern 'datasets/AGRI_processed/preprocessed_video/*_agri_video_*.npz' \
      --checkpoint checkpoints/satellite_tokenizer/agri/agri_training_xxx/checkpoints/iter_000240000_ema.jit \
      --region 0:256,0:256 \
      --max_samples 100 \
      --save_reconstructions true
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_pattern) DATA_PATTERN="$2"; shift 2 ;;
        --data_pattern=*) DATA_PATTERN="${1#*=}"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
        --checkpoint_enc) CHECKPOINT_ENC="$2"; shift 2 ;;
        --checkpoint_enc=*) CHECKPOINT_ENC="${1#*=}"; shift ;;
        --checkpoint_dec) CHECKPOINT_DEC="$2"; shift 2 ;;
        --checkpoint_dec=*) CHECKPOINT_DEC="${1#*=}"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --output_dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --region) REGION="$2"; shift 2 ;;
        --region=*) REGION="${1#*=}"; shift ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        --max_samples=*) MAX_SAMPLES="${1#*=}"; shift ;;
        --cuda_visible_devices) CUDA_VISIBLE_DEVICES_DEFAULT="$2"; shift 2 ;;
        --cuda_visible_devices=*) CUDA_VISIBLE_DEVICES_DEFAULT="${1#*=}"; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --device=*) DEVICE="${1#*=}"; shift ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --dtype=*) DTYPE="${1#*=}"; shift ;;
        --use_mask_channel) USE_MASK_CHANNEL="$2"; shift 2 ;;
        --use_mask_channel=*) USE_MASK_CHANNEL="${1#*=}"; shift ;;
        --mask_metrics) MASK_METRICS="$2"; shift 2 ;;
        --mask_metrics=*) MASK_METRICS="${1#*=}"; shift ;;
        --evaluate_mask_channel) EVALUATE_MASK_CHANNEL="$2"; shift 2 ;;
        --evaluate_mask_channel=*) EVALUATE_MASK_CHANNEL="${1#*=}"; shift ;;
        --save_reconstructions) SAVE_RECONSTRUCTIONS="$2"; shift 2 ;;
        --save_reconstructions=*) SAVE_RECONSTRUCTIONS="${1#*=}"; shift ;;
        --data_key) DATA_KEY="$2"; shift 2 ;;
        --data_key=*) DATA_KEY="${1#*=}"; shift ;;
        --mask_key) MASK_KEY="$2"; shift 2 ;;
        --mask_key=*) MASK_KEY="${1#*=}"; shift ;;
        --python_bin) PYTHON_BIN="$2"; shift 2 ;;
        --python_bin=*) PYTHON_BIN="${1#*=}"; shift ;;
        --project_root) PROJECT_ROOT="$2"; shift 2 ;;
        --project_root=*) PROJECT_ROOT="${1#*=}"; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "未知参数: $1"; print_help; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" && ( -z "$CHECKPOINT_ENC" || -z "$CHECKPOINT_DEC" ) ]]; then
    echo "错误: 需要提供 --checkpoint，或同时提供 --checkpoint_enc 和 --checkpoint_dec"
    exit 1
fi

if [[ -n "$CONDA_ENV_PATH" ]]; then
    if [[ ! -f "${CONDA_ENV_PATH}/etc/profile.d/conda.sh" ]]; then
        echo "错误: CONDA_ENV_PATH 无效: $CONDA_ENV_PATH"
        exit 1
    fi
    source "${CONDA_ENV_PATH}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
fi

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_DEFAULT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_DIR" logs
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="logs/evaluate_agri_tokenizer_region_${TIMESTAMP}.log"

CMD=(
    "$PYTHON_BIN" -m cosmos_predict1.tokenizer_satellite.inference.evaluate_agri_tokenizer_region
    --data_pattern "$DATA_PATTERN"
    --output_dir "$OUTPUT_DIR"
    --data_key "$DATA_KEY"
    --mask_key "$MASK_KEY"
    --use_mask_channel "$USE_MASK_CHANNEL"
    --device "$DEVICE"
    --dtype "$DTYPE"
)

if [[ -n "$CHECKPOINT" ]]; then
    CMD+=(--checkpoint "$CHECKPOINT")
else
    CMD+=(--checkpoint_enc "$CHECKPOINT_ENC" --checkpoint_dec "$CHECKPOINT_DEC")
fi

if [[ -n "$REGION" ]]; then
    CMD+=(--region "$REGION")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    CMD+=(--max_samples "$MAX_SAMPLES")
fi

case "${MASK_METRICS,,}" in
    true|1|yes|y) CMD+=(--mask_metrics) ;;
    false|0|no|n) CMD+=(--no_mask_metrics) ;;
    *) echo "错误: --mask_metrics 只能是 true/false"; exit 1 ;;
esac

case "${EVALUATE_MASK_CHANNEL,,}" in
    true|1|yes|y) CMD+=(--evaluate_mask_channel) ;;
    false|0|no|n) ;;
    *) echo "错误: --evaluate_mask_channel 只能是 true/false"; exit 1 ;;
esac

case "${SAVE_RECONSTRUCTIONS,,}" in
    true|1|yes|y) CMD+=(--save_reconstructions) ;;
    false|0|no|n) ;;
    *) echo "错误: --save_reconstructions 只能是 true/false"; exit 1 ;;
esac

echo "=========================================="
echo "AGRI Tokenizer 区域批量推理与评估"
echo "项目根目录: $PROJECT_ROOT"
echo "数据路径: $DATA_PATTERN"
echo "输出目录: $OUTPUT_DIR"
echo "区域: ${REGION:-full}"
echo "日志文件: $LOG_FILE"
echo "=========================================="
printf '执行命令:'
printf ' %q' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
