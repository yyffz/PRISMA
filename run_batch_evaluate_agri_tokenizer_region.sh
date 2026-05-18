#!/bin/bash

# 在新裁剪区域上批量评估 AGRI 视频 Tokenizer（微调 vs 预训练）
# 裁剪区域与 prepare_agri_video_tokens 一致：h=[300:1500], w=[2200:3400]，5 时刻 ref±45/30/15/0/15min
# 数据源：AGRI_processed 根目录下的原始 NPZ（YYYYMMDDHHMM_agri.npz）
# 使用方法: ./run_batch_evaluate_agri_tokenizer_region.sh [选项]

export CUDA_VISIBLE_DEVICES=5

# 默认参数
START="2025-07-01"
END="2025-07-03"
INPUT_DIR="/public/share/users/sunhaofei/yyf_data/AGRI_processed"
PRETRAINED_DIR="/public/home/sunhaofei/cosmos-predict1/checkpoints/Cosmos-Tokenize1-CV4x8x8-360p"
FINETUNED_DIR="/public/home/sunhaofei/cosmos-predict1/checkpoints/satellite_tokenizer/agri/agri_training_2026-02-08_13-37-55/checkpoints"
FINETUNED_ITER=210000
OUTPUT_DIR="outputs/evaluation_results_agri_region"
DEVICE="cuda"
DTYPE="bfloat16"
MAX_SAMPLES=""
STEP_MINUTES=30
STATS_PATH="/public/home/sunhaofei/yyf/DGPR/channel_stats_agri_15ch.pth"
MIN_VALID_FRAMES=5
MIN_VALID_RATIO=0.5
SAVE_PLOTS=true
PLOT_SAMPLES=5
REPORT_BOTH_METRICS=true

# Conda 环境
CONDA_ENV_PATH="/public/home/sunhaofei/anaconda3"
CONDA_ENV_NAME="cosmos-predict1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/batch_evaluate_agri_tokenizer_region.py"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --start)
            START="$2"
            shift 2
            ;;
        --end)
            END="$2"
            shift 2
            ;;
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --pretrained_dir)
            PRETRAINED_DIR="$2"
            shift 2
            ;;
        --finetuned_dir)
            FINETUNED_DIR="$2"
            shift 2
            ;;
        --finetuned_iter)
            FINETUNED_ITER="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --step_minutes)
            STEP_MINUTES="$2"
            shift 2
            ;;
        --stats_path)
            STATS_PATH="$2"
            shift 2
            ;;
        --no_mask_channel)
            NO_MASK_CHANNEL="--no_mask_channel"
            shift
            ;;
        --save_plots)
            SAVE_PLOTS=true
            shift
            ;;
        --plot_samples)
            PLOT_SAMPLES="$2"
            shift 2
            ;;
        --min_valid_frames)
            MIN_VALID_FRAMES="$2"
            shift 2
            ;;
        --min_valid_ratio)
            MIN_VALID_RATIO="$2"
            shift 2
            ;;
        --report_both_metrics)
            REPORT_BOTH_METRICS=true
            shift
            ;;
        -h|--help)
            echo "在新裁剪区域上批量评估 AGRI 视频 Tokenizer"
            echo ""
            echo "裁剪区域: h=[300:1500], w=[2200:3400] (1200×1200)"
            echo "5 个时刻: ref-45min, ref-30min, ref-15min, ref, ref+15min"
            echo "数据: INPUT_DIR 下 YYYYMMDDHHMM_agri.npz"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "可选参数:"
            echo "  --start DATE             开始时间 (如 2024-07-01)"
            echo "  --end DATE               结束时间 (如 2024-07-02)"
            echo "  --input_dir PATH         AGRI_processed 根目录 (默认: .../AGRI_processed)"
            echo "  --pretrained_dir PATH    预训练模型目录"
            echo "  --finetuned_dir PATH     微调模型 checkpoint 目录"
            echo "  --finetuned_iter INT     微调模型迭代次数 (默认: 210000)"
            echo "  --output_dir PATH        输出目录 (默认: outputs/evaluation_results_agri_region)"
            echo "  --device DEVICE          计算设备 (默认: cuda)"
            echo "  --dtype DTYPE            数据类型 (默认: bfloat16)"
            echo "  --max_samples INT        最大样本数（用于测试）"
            echo "  --step_minutes INT       参考时间步长分钟 (默认: 15)"
            echo "  --stats_path PATH        通道统计量 .pth"
            echo "  --no_mask_channel        微调模型不使用 mask 通道"
            echo "  --save_plots              对部分样本保存 3 通道对比图"
            echo "  --plot_samples INT        绘制对比图的样本数 (默认: 5)"
            echo "  --min_valid_frames INT    5 帧中至少几帧有效才纳入评估 (默认: 2)"
            echo "  --min_valid_ratio FLOAT   单帧有效比例阈值 (默认: 0.1)"
            echo "  --report_both_metrics     同时输出「仅有效区」与「全图（含无效区）」指标"
            echo ""
            echo "示例:"
            echo "  $0 --start 2024-07-01 --end 2024-07-03"
            echo "  $0 --start 2024-07-01 --end 2024-07-02 --max_samples 20"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi
if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "错误: 预训练模型目录不存在: $PRETRAINED_DIR"
    exit 1
fi
if [ ! -d "$FINETUNED_DIR" ]; then
    echo "错误: 微调模型目录不存在: $FINETUNED_DIR"
    exit 1
fi
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "错误: 评估脚本不存在: $EVAL_SCRIPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/batch_evaluation_region_${TIMESTAMP}.log"

echo "=========================================="
echo "AGRI Tokenizer 批量评估（新裁剪区域）"
echo "=========================================="
echo "裁剪区域: h=[300:1500], w=[2200:3400] (1200×1200)"
echo "时间范围: $START ~ $END  步长: ${STEP_MINUTES} 分钟"
echo "输入目录: $INPUT_DIR"
echo "预训练模型: $PRETRAINED_DIR"
echo "微调模型: $FINETUNED_DIR  iter=$FINETUNED_ITER"
echo "输出目录: $OUTPUT_DIR"
if [ "$SAVE_PLOTS" = true ]; then
    echo "保存对比图: 是 (最多 $PLOT_SAMPLES 个样本)"
fi
echo "异常样本过滤: 至少 ${MIN_VALID_FRAMES} 帧 valid_ratio >= ${MIN_VALID_RATIO}"
echo "日志: $LOG_FILE"
echo "=========================================="

# 激活 conda
if [ -d "$CONDA_ENV_PATH" ]; then
    source "${CONDA_ENV_PATH}/bin/activate"
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
fi

# 构建命令
CMD="python $EVAL_SCRIPT --start \"$START\" --end \"$END\" --input_dir \"$INPUT_DIR\" \
    --pretrained_dir \"$PRETRAINED_DIR\" --finetuned_dir \"$FINETUNED_DIR\" --finetuned_iter $FINETUNED_ITER \
    --output_dir \"$OUTPUT_DIR\" --device $DEVICE --dtype $DTYPE --step_minutes $STEP_MINUTES \
    --stats_path \"$STATS_PATH\" --min_valid_frames $MIN_VALID_FRAMES --min_valid_ratio $MIN_VALID_RATIO $NO_MASK_CHANNEL"
[ -n "$MAX_SAMPLES" ] && CMD="$CMD --max_samples $MAX_SAMPLES"
[ "$SAVE_PLOTS" = true ] && CMD="$CMD --save_plots --plot_samples $PLOT_SAMPLES"
[ "$REPORT_BOTH_METRICS" = true ] && CMD="$CMD --report_both_metrics"

echo ""
echo "执行: $CMD"
eval $CMD 2>&1 | tee "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "批量评估完成。输出: $OUTPUT_DIR  日志: $LOG_FILE"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "批量评估失败，请查看: $LOG_FILE"
    echo "=========================================="
    exit 1
fi
