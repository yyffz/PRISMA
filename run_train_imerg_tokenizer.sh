#!/bin/bash

# IMERG Tokenizer 训练任务提交脚本
# 使用方法: ./run_train_imerg_tokenizer.sh [选项]
#
# IMERG数据是单通道降水数据，在dataloader中会被扩展为3通道以匹配预训练模型
# 可选添加QI（质量指数）通道作为第4通道
export CUDA_VISIBLE_DEVICES=0,1 
# 自动检测项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# 默认参数
DATA_PATTERN="/public/share/users/sunhaofei/imerg_processed_new/*_imerg_*.npz"
USE_QI_CHANNEL=false  # true=4通道（3个扩展通道+1个QI通道），false=3通道（仅扩展通道）
NUM_GPUS=2  # 使用的GPU数量
BATCH_SIZE=8  # 批次大小（减小以降低内存占用）
NUM_WORKERS=8  # 数据加载worker数量（减小以降低内存占用）
PREFETCH_FACTOR=4  # 预取因子（减小以降低内存占用）
LEARNING_RATE=5e-5  # 学习率
WARMUP_STEPS=10000  # Warmup步数
MAX_ITER=500000  # 最大迭代次数
CHECKPOINT_INTERVAL=10000  # Checkpoint保存间隔
VAL_INTERVAL=5000  # 验证间隔
RESUME_FROM_CHECKPOINT=""  # 断点续训checkpoint路径
LOAD_TRAINING_STATE=true  # 是否加载训练状态

# Conda环境路径
CONDA_ENV_PATH="/public/home/sunhaofei/anaconda3"
CONDA_ENV_NAME="cosmos-predict1"

# 训练脚本路径
TRAIN_SCRIPT="${PROJECT_ROOT}/cosmos_predict1/tokenizer_satellite/training/train.py"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_pattern)
            DATA_PATTERN="$2"
            shift 2
            ;;
        --use_qi_channel)
            USE_QI_CHANNEL="$2"
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
            shift 2
            ;;
        -h|--help)
            echo "IMERG Tokenizer 训练任务提交脚本"
            echo ""
            echo "使用方法: $0 [选项]"
            echo ""
            echo "IMERG数据说明："
            echo "  - IMERG是单通道降水数据（1通道）"
            echo "  - 在dataloader中会被扩展为3通道以匹配预训练模型"
            echo "  - 可选添加QI（质量指数）通道作为第4通道"
            echo ""
            echo "可选参数:"
            echo "  --data_pattern PATTERN      数据文件路径模式 (默认: /public/share/users/sunhaofei/imerg_processed/*_imerg_*.npz)"
            echo "  --use_qi_channel BOOL       是否使用QI通道 (true/false, 默认: true)"
            echo "                              true=4通道（3个扩展通道+1个QI通道）"
            echo "                              false=3通道（仅扩展通道）"
            echo "  --num_gpus INT              使用的GPU数量 (默认: 1)"
            echo "  --batch_size INT            批次大小 (默认: 8)"
            echo "  --num_workers INT           数据加载worker数量 (默认: 8)"
            echo "  --prefetch_factor INT       预取因子 (默认: 4)"
            echo "  --learning_rate FLOAT       学习率 (默认: 5e-5)"
            echo "  --warmup_steps INT          Warmup步数 (默认: 10000)"
            echo "  --max_iter INT              最大迭代次数 (默认: 500000)"
            echo "  --checkpoint_interval INT   Checkpoint保存间隔 (默认: 10000)"
            echo "  --val_interval INT          验证间隔 (默认: 5000)"
            echo "  --resume_from_checkpoint PATH 断点续训checkpoint路径（可选）"
            echo "  --load_training_state BOOL  是否加载训练状态 (默认: true)"
            echo "  --project_root PATH         项目根目录"
            echo ""
            echo "示例:"
            echo "  # 使用QI通道训练（默认，4通道）"
            echo "  $0 --num_gpus=1 --batch_size=8"
            echo ""
            echo "  # 不使用QI通道训练（3通道）"
            echo "  $0 --use_qi_channel=false --num_gpus=2"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查项目根目录是否存在
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误: 项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

# 检查训练脚本是否存在
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "错误: 训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# 检查数据路径
if [[ "$DATA_PATTERN" == /* ]]; then
    FULL_DATA_PATTERN="$DATA_PATTERN"
else
    FULL_DATA_PATTERN="${PROJECT_ROOT}/${DATA_PATTERN}"
fi

FIRST_FILE=$(compgen -G "${FULL_DATA_PATTERN}" 2>/dev/null | head -n 1)

if [ -z "$FIRST_FILE" ]; then
    echo "警告: 未找到匹配的数据文件: $FULL_DATA_PATTERN"
    echo "请检查数据路径是否正确"
    if [[ "$FULL_DATA_PATTERN" == *"*"* ]]; then
        DATA_DIR_BASE=$(dirname "${FULL_DATA_PATTERN%%\**}")
    else
        DATA_DIR_BASE=$(dirname "$FULL_DATA_PATTERN")
    fi
    if [ -d "$DATA_DIR_BASE" ]; then
        echo "目录 $DATA_DIR_BASE 存在，但未找到匹配的文件"
        echo "目录中的文件示例："
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

# 根据use_qi_channel设置通道数
# IMERG: 1通道 -> 扩展为3通道 -> 可选+1个QI通道
EXPAND_CHANNELS=3
if [ "$USE_QI_CHANNEL" = "true" ] || [ "$USE_QI_CHANNEL" = "True" ] || [ "$USE_QI_CHANNEL" = "1" ]; then
    NUM_CHANNELS=4  # 3个扩展通道 + 1个QI通道
    INCLUDE_QI_CHANNEL=true
    USE_QI_FLAG="true"
else
    NUM_CHANNELS=3  # 仅3个扩展通道
    INCLUDE_QI_CHANNEL=false
    USE_QI_FLAG="false"
fi

# 生成日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_imerg_${TIMESTAMP}.log"

echo "=========================================="
echo "IMERG Tokenizer 训练任务"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "数据路径: $DATA_PATTERN"
echo "通道配置:"
echo "  - 原始通道: 1（降水）"
echo "  - 扩展通道: $EXPAND_CHANNELS"
echo "  - 使用QI通道: $USE_QI_FLAG"
echo "  - 最终通道数: $NUM_CHANNELS"
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
    echo "断点续训: 自动检测"
fi
echo "日志文件: $LOG_FILE"
echo "=========================================="

# 激活conda环境
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "激活conda环境: $CONDA_ENV_NAME"
    source "${CONDA_ENV_PATH}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    if [ $? -ne 0 ]; then
        echo "错误: 无法激活conda环境: $CONDA_ENV_NAME"
        exit 1
    fi
    echo "Conda环境已激活"
else
    echo "警告: Conda环境路径不存在，跳过环境激活"
fi

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 构建训练命令
TRAIN_CMD="torchrun --nproc_per_node=${NUM_GPUS} -m cosmos_predict1.tokenizer_satellite.training.train"
TRAIN_CMD="${TRAIN_CMD} --config=cosmos_predict1/tokenizer_satellite/training/configs/config.py"
TRAIN_CMD="${TRAIN_CMD} --"
TRAIN_CMD="${TRAIN_CMD} experiment=imerg_training"
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

# 断点续训配置
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    if [[ "$RESUME_FROM_CHECKPOINT" == /* ]]; then
        FULL_CHECKPOINT_PATH="$RESUME_FROM_CHECKPOINT"
    else
        FULL_CHECKPOINT_PATH="${PROJECT_ROOT}/${RESUME_FROM_CHECKPOINT}"
    fi
    
    if [ ! -f "$FULL_CHECKPOINT_PATH" ]; then
        echo "警告: 指定的checkpoint文件不存在: $FULL_CHECKPOINT_PATH"
        echo "将尝试自动从latest_checkpoint.txt恢复"
    else
        echo "使用指定的checkpoint: $FULL_CHECKPOINT_PATH"
        TRAIN_CMD="${TRAIN_CMD} checkpoint.load_path=\"${RESUME_FROM_CHECKPOINT}\""
        
        if [ "$LOAD_TRAINING_STATE" = "true" ] || [ "$LOAD_TRAINING_STATE" = "True" ] || [ "$LOAD_TRAINING_STATE" = "1" ]; then
            TRAIN_CMD="${TRAIN_CMD} checkpoint.load_training_state=true"
            echo "将加载训练状态（优化器、调度器等）"
        else
            TRAIN_CMD="${TRAIN_CMD} checkpoint.load_training_state=false"
            echo "只加载模型权重，不加载训练状态"
        fi
    fi
else
    echo "未指定checkpoint路径，将自动检测latest_checkpoint.txt或使用预训练权重"
fi

# 根据是否使用QI通道添加相应参数
if [ "$USE_QI_FLAG" = "false" ]; then
    echo "配置: 不使用QI通道（3通道）"
    TRAIN_CMD="${TRAIN_CMD} model.config.network.in_channels=3"
    TRAIN_CMD="${TRAIN_CMD} model.config.network.out_channels=3"
    TRAIN_CMD="${TRAIN_CMD} dataloader_train.dataset.target_channels=3"
    TRAIN_CMD="${TRAIN_CMD} dataloader_train.dataset.include_mask_channel=false"
    TRAIN_CMD="${TRAIN_CMD} dataloader_val.dataset.target_channels=3"
    TRAIN_CMD="${TRAIN_CMD} dataloader_val.dataset.include_mask_channel=false"
    TRAIN_CMD="${TRAIN_CMD} checkpoint.jit.input_shape=[1,3,256,256]"
else
    echo "配置: 使用QI通道（4通道）"
fi

# 打印完整命令
echo ""
echo "执行命令:"
echo "$TRAIN_CMD"
echo ""
echo "开始训练..."
echo "日志将保存到: $LOG_FILE"
echo ""

# 执行训练命令
eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"

# 检查训练是否成功
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
