#!/bin/bash

# DPR Ka Tokenizer 微调任务提交脚本
# 输入通道: [组合反射率, 组合反射率, mask]
# 使用方法: ./run_train_dpr_ka_tokenizer.sh [选项]

export CUDA_VISIBLE_DEVICES=1,2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

DATA_PATTERN="/public/share/users/sunhaofei/yyf_data/DPR_Ka_composite_crops_2024_full_new/*_DPR_Ka.npz"
NUM_GPUS=2
BATCH_SIZE=8
NUM_WORKERS=2
PREFETCH_FACTOR=4
LEARNING_RATE=5e-5
WARMUP_STEPS=10000
MAX_ITER=500000
CHECKPOINT_INTERVAL=10000
VAL_INTERVAL=5000
RESUME_FROM_CHECKPOINT=""
LOAD_TRAINING_STATE=false

CONDA_ENV_PATH="/public/home/sunhaofei/anaconda3"
CONDA_ENV_NAME="cosmos-predict1"

TRAIN_SCRIPT="${PROJECT_ROOT}/cosmos_predict1/tokenizer_satellite/training/train.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_pattern)
            DATA_PATTERN="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --prefetch_factor)
            PREFETCH_FACTOR="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --warmup_steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --max_iter)
            MAX_ITER="$2"
            shift 2
            ;;
        --checkpoint_interval)
            CHECKPOINT_INTERVAL="$2"
            shift 2
            ;;
        --val_interval)
            VAL_INTERVAL="$2"
            shift 2
            ;;
        --resume_from_checkpoint)
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
            ;;
        --load_training_state)
            LOAD_TRAINING_STATE="$2"
            shift 2
            ;;
        --project_root)
            PROJECT_ROOT="$2"
            TRAIN_SCRIPT="${PROJECT_ROOT}/cosmos_predict1/tokenizer_satellite/training/train.py"
            shift 2
            ;;
        -h|--help)
            echo "DPR Ka Tokenizer 微调任务提交脚本"
            echo ""
            echo "使用方法: $0 [选项]"
            echo ""
            echo "可选参数:"
            echo "  --data_pattern PATTERN        数据文件路径模式"
            echo "  --num_gpus INT                使用的GPU数量 (默认: 1)"
            echo "  --batch_size INT              批次大小 (默认: 8)"
            echo "  --num_workers INT             数据加载worker数量 (默认: 8)"
            echo "  --prefetch_factor INT         预取因子 (默认: 4)"
            echo "  --learning_rate FLOAT         学习率 (默认: 5e-5)"
            echo "  --warmup_steps INT            Warmup步数 (默认: 10000)"
            echo "  --max_iter INT                最大迭代次数 (默认: 500000)"
            echo "  --checkpoint_interval INT     Checkpoint保存间隔 (默认: 10000)"
            echo "  --val_interval INT            验证间隔 (默认: 5000)"
            echo "  --resume_from_checkpoint PATH 指定checkpoint路径；为空则加载预训练模型"
            echo "  --load_training_state BOOL    是否加载优化器/调度器等训练状态 (默认: false)"
            echo "  --project_root PATH           项目根目录"
            echo ""
            echo "示例:"
            echo "  $0 --num_gpus 4 --batch_size 16"
            echo "  $0 --resume_from_checkpoint checkpoints/satellite_tokenizer/dpr_ka/.../iter_000100000.pt --load_training_state true"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误: 项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "错误: 训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

if [[ "$DATA_PATTERN" == /* ]]; then
    FULL_DATA_PATTERN="$DATA_PATTERN"
else
    FULL_DATA_PATTERN="${PROJECT_ROOT}/${DATA_PATTERN}"
fi

FIRST_FILE=$(compgen -G "${FULL_DATA_PATTERN}" 2>/dev/null | head -n 1)
if [ -z "$FIRST_FILE" ]; then
    echo "警告: 未找到匹配的数据文件: $FULL_DATA_PATTERN"
    if [[ "$FULL_DATA_PATTERN" == *"*"* ]]; then
        DATA_DIR_BASE=$(dirname "${FULL_DATA_PATTERN%%\**}")
    else
        DATA_DIR_BASE=$(dirname "$FULL_DATA_PATTERN")
    fi
    if [ -d "$DATA_DIR_BASE" ]; then
        echo "目录 $DATA_DIR_BASE 存在，但未找到匹配的文件"
        ls -1 "$DATA_DIR_BASE" 2>/dev/null | head -n 5
    else
        echo "目录 $DATA_DIR_BASE 不存在"
    fi
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "找到数据文件: $FIRST_FILE"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_dpr_ka_${TIMESTAMP}.log"

echo "=========================================="
echo "DPR Ka Tokenizer 微调任务"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "数据路径: $DATA_PATTERN"
echo "输入通道: 3 ([组合反射率, 组合反射率, mask])"
echo "GPU数量: $NUM_GPUS"
echo "批次大小: $BATCH_SIZE"
echo "Worker数量: $NUM_WORKERS"
echo "预取因子: $PREFETCH_FACTOR"
echo "学习率: $LEARNING_RATE"
echo "Warmup步数: $WARMUP_STEPS"
echo "最大迭代次数: $MAX_ITER"
echo "Checkpoint间隔: $CHECKPOINT_INTERVAL"
echo "验证间隔: $VAL_INTERVAL"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "断点续训: 从 $RESUME_FROM_CHECKPOINT 恢复"
    echo "加载训练状态: $LOAD_TRAINING_STATE"
else
    echo "微调初始化: 使用配置中的预训练模型"
fi
echo "日志文件: $LOG_FILE"
echo "=========================================="

if [ -d "$CONDA_ENV_PATH" ]; then
    echo "激活conda环境: $CONDA_ENV_NAME"
    source "${CONDA_ENV_PATH}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    if [ $? -ne 0 ]; then
        echo "错误: 无法激活conda环境: $CONDA_ENV_NAME"
        exit 1
    fi
else
    echo "警告: Conda环境路径不存在，跳过环境激活"
fi

cd "$PROJECT_ROOT" || exit 1

MASTER_PORT=$((29500 + RANDOM % 500))
echo "使用分布式训练端口: $MASTER_PORT"

TRAIN_CMD="torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} -m cosmos_predict1.tokenizer_satellite.training.train"
TRAIN_CMD="${TRAIN_CMD} --config=cosmos_predict1/tokenizer_satellite/training/configs/config.py"
TRAIN_CMD="${TRAIN_CMD} --"
TRAIN_CMD="${TRAIN_CMD} experiment=dpr_ka_training"
TRAIN_CMD="${TRAIN_CMD} dataloader_train.dataset.data_pattern=\"${DATA_PATTERN}\""
TRAIN_CMD="${TRAIN_CMD} dataloader_train.batch_size=${BATCH_SIZE}"
TRAIN_CMD="${TRAIN_CMD} dataloader_train.num_workers=${NUM_WORKERS}"
TRAIN_CMD="${TRAIN_CMD} dataloader_train.prefetch_factor=${PREFETCH_FACTOR}"
TRAIN_CMD="${TRAIN_CMD} dataloader_val.dataset.data_pattern=\"${DATA_PATTERN}\""
TRAIN_CMD="${TRAIN_CMD} dataloader_val.batch_size=${BATCH_SIZE}"
TRAIN_CMD="${TRAIN_CMD} dataloader_val.num_workers=${NUM_WORKERS}"
TRAIN_CMD="${TRAIN_CMD} dataloader_val.prefetch_factor=${PREFETCH_FACTOR}"
TRAIN_CMD="${TRAIN_CMD} optimizer.lr=${LEARNING_RATE}"
TRAIN_CMD="${TRAIN_CMD} scheduler.warmup=${WARMUP_STEPS}"
TRAIN_CMD="${TRAIN_CMD} trainer.max_iter=${MAX_ITER}"
TRAIN_CMD="${TRAIN_CMD} trainer.validation_iter=${VAL_INTERVAL}"
TRAIN_CMD="${TRAIN_CMD} checkpoint.save_iter=${CHECKPOINT_INTERVAL}"
TRAIN_CMD="${TRAIN_CMD} model.config.network.in_channels=3"
TRAIN_CMD="${TRAIN_CMD} model.config.network.out_channels=3"
TRAIN_CMD="${TRAIN_CMD} checkpoint.jit.input_shape=[1,3,256,256]"

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    if [[ "$RESUME_FROM_CHECKPOINT" == /* ]]; then
        FULL_CHECKPOINT_PATH="$RESUME_FROM_CHECKPOINT"
    else
        FULL_CHECKPOINT_PATH="${PROJECT_ROOT}/${RESUME_FROM_CHECKPOINT}"
    fi

    if [ ! -f "$FULL_CHECKPOINT_PATH" ]; then
        echo "警告: 指定的checkpoint文件不存在: $FULL_CHECKPOINT_PATH"
        echo "将使用配置中的预训练模型初始化"
    else
        TRAIN_CMD="${TRAIN_CMD} checkpoint.load_path=\"${RESUME_FROM_CHECKPOINT}\""
        if [ "$LOAD_TRAINING_STATE" = "true" ] || [ "$LOAD_TRAINING_STATE" = "True" ] || [ "$LOAD_TRAINING_STATE" = "1" ]; then
            TRAIN_CMD="${TRAIN_CMD} checkpoint.load_training_state=true"
        else
            TRAIN_CMD="${TRAIN_CMD} checkpoint.load_training_state=false"
        fi
    fi
fi

echo ""
echo "执行命令:"
echo "$TRAIN_CMD"
echo ""
echo "开始训练..."
echo "日志将保存到: $LOG_FILE"
echo ""

eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "训练任务完成！"
    echo "日志文件: $LOG_FILE"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "训练任务失败，退出代码: $TRAIN_EXIT_CODE"
    echo "请查看日志文件: $LOG_FILE"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi
