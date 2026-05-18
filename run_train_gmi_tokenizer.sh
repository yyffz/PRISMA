#!/bin/bash

# GMI Tokenizer 训练任务提交脚本
# 使用方法: ./run_train_gmi_tokenizer.sh [选项]
export CUDA_VISIBLE_DEVICES=1
# 自动检测项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# 默认参数
DATA_PATTERN="datasets/GMI_processed/preprocessed/*_GMI.npz"
USE_MASK_CHANNEL=true  # true=14通道（13个GMI通道+1个mask通道），false=13通道（仅GMI通道）
NUM_GPUS=1  # 使用的GPU数量
BATCH_SIZE=8  # 批次大小
NUM_WORKERS=8  # 数据加载worker数量
PREFETCH_FACTOR=4  # 预取因子
LEARNING_RATE=5e-5  # 学习率
WARMUP_STEPS=10000  # Warmup步数
MAX_ITER=500000  # 最大迭代次数
CHECKPOINT_INTERVAL=10000  # Checkpoint保存间隔
VAL_INTERVAL=5000  # 验证间隔
RESUME_FROM_CHECKPOINT="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/yangyunfan/cosmos-predict1-main/checkpoints/satellite_tokenizer/gmi/gmi_training_2025-12-28_13-52-47/checkpoints/iter_000220000.pt"  # 断点续训：指定checkpoint路径（如：checkpoints/satellite_tokenizer/gmi/gmi_training_2026-01-05_12-49-28/checkpoints/iter_000240000.pt），如果为空则自动从latest_checkpoint.txt恢复
LOAD_TRAINING_STATE=true  # 是否加载训练状态（优化器、调度器等），用于断点续训

# Conda环境路径
CONDA_ENV_PATH="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/env/Miniforge3"
CONDA_ENV_NAME="cosmos_predict1"

# 训练脚本路径（相对于项目根目录）
TRAIN_SCRIPT="${PROJECT_ROOT}/cosmos_predict1/tokenizer_satellite/training/train.py"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_pattern)
            DATA_PATTERN="$2"
            shift 2
            ;;
        --use_mask_channel)
            USE_MASK_CHANNEL="$2"
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
            echo "GMI Tokenizer 训练任务提交脚本"
            echo ""
            echo "使用方法: $0 [选项]"
            echo ""
            echo "必需参数:"
            echo "  无（所有参数都有默认值）"
            echo ""
            echo "可选参数:"
            echo "  --data_pattern PATTERN      数据文件路径模式 (默认: datasets/GMI_processed/preprocessed/*_GMI.npz)"
            echo "  --use_mask_channel BOOL     是否使用mask通道 (true/false, 默认: true)"
            echo "                              true=14通道（13个GMI通道+1个mask通道）"
            echo "                              false=13通道（仅GMI通道）"
            echo "  --num_gpus INT              使用的GPU数量 (默认: 1)"
            echo "  --batch_size INT            批次大小 (默认: 8)"
            echo "  --num_workers INT            数据加载worker数量 (默认: 8)"
            echo "  --prefetch_factor INT       预取因子 (默认: 4)"
            echo "  --learning_rate FLOAT       学习率 (默认: 5e-5)"
            echo "  --warmup_steps INT          Warmup步数 (默认: 10000)"
            echo "  --max_iter INT              最大迭代次数 (默认: 500000)"
            echo "  --checkpoint_interval INT    Checkpoint保存间隔 (默认: 10000)"
            echo "  --val_interval INT           验证间隔 (默认: 5000)"
            echo "  --resume_from_checkpoint PATH 断点续训：指定checkpoint路径（可选）"
            echo "                              如果为空，会自动从latest_checkpoint.txt恢复"
            echo "  --load_training_state BOOL   是否加载训练状态（优化器、调度器等）(默认: true)"
            echo "  --project_root PATH         项目根目录 (默认: /cpfs01/.../cosmos-predict1-main)"
            echo ""
            echo "示例:"
            echo "  # 使用mask通道训练（默认）"
            echo "  $0 --num_gpus=4 --batch_size=16"
            echo ""
            echo "  # 不使用mask通道训练"
            echo "  $0 --use_mask_channel=false --num_gpus=2"
            echo ""
            echo "  # 自定义数据路径和训练参数"
            echo "  $0 --data_pattern=\"datasets/GMI_processed/preprocessed/*_GMI.npz\" \\"
            echo "      --use_mask_channel=true \\"
            echo "      --num_gpus=4 \\"
            echo "      --batch_size=16 \\"
            echo "      --learning_rate=1e-4"
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

# 检查数据路径是否存在
# 处理相对路径和绝对路径
if [[ "$DATA_PATTERN" == /* ]]; then
    # 绝对路径
    FULL_DATA_PATTERN="$DATA_PATTERN"
else
    # 相对路径，需要转换为绝对路径
    FULL_DATA_PATTERN="${PROJECT_ROOT}/${DATA_PATTERN}"
fi

# 验证数据文件是否存在（检查前几个文件）
# 使用compgen -G来检查通配符匹配，这比ls更可靠
FIRST_FILE=$(compgen -G "${FULL_DATA_PATTERN}" 2>/dev/null | head -n 1)

if [ -z "$FIRST_FILE" ]; then
    echo "警告: 未找到匹配的数据文件: $FULL_DATA_PATTERN"
    echo "请检查数据路径是否正确"
    # 尝试列出目录内容以帮助调试
    # 提取目录部分（去除通配符部分）
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

# 根据use_mask_channel设置通道数
if [ "$USE_MASK_CHANNEL" = "true" ] || [ "$USE_MASK_CHANNEL" = "True" ] || [ "$USE_MASK_CHANNEL" = "1" ]; then
    NUM_CHANNELS=14
    INCLUDE_MASK_CHANNEL=true
    USE_MASK_FLAG="true"
else
    NUM_CHANNELS=13
    INCLUDE_MASK_CHANNEL=false
    USE_MASK_FLAG="false"
fi

# 生成日志文件名（基于当前时间）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_gmi_${TIMESTAMP}.log"

echo "=========================================="
echo "GMI Tokenizer 训练任务"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "数据路径: $DATA_PATTERN"
echo "使用mask通道: $USE_MASK_FLAG ($NUM_CHANNELS 通道)"
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
    echo "断点续训: 自动检测（如果存在latest_checkpoint.txt）"
fi
echo "日志文件: $LOG_FILE"
echo "=========================================="

# 激活conda环境（如果存在）
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
# 使用torchrun进行分布式训练
# 注意：--config 使用文件路径格式（相对项目根目录），不是模块路径
# 生成随机端口 (29500-29999) 以避免端口冲突
MASTER_PORT=$((29500 + RANDOM % 500))
echo "使用分布式训练端口: $MASTER_PORT"
TRAIN_CMD="torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} -m cosmos_predict1.tokenizer_satellite.training.train"
TRAIN_CMD="${TRAIN_CMD} --config=cosmos_predict1/tokenizer_satellite/training/configs/config.py"
TRAIN_CMD="${TRAIN_CMD} --"
TRAIN_CMD="${TRAIN_CMD} experiment=gmi_training"
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
    # 如果指定了checkpoint路径，使用该路径
    # 处理相对路径和绝对路径
    if [[ "$RESUME_FROM_CHECKPOINT" == /* ]]; then
        # 绝对路径
        FULL_CHECKPOINT_PATH="$RESUME_FROM_CHECKPOINT"
    else
        # 相对路径，转换为绝对路径
        FULL_CHECKPOINT_PATH="${PROJECT_ROOT}/${RESUME_FROM_CHECKPOINT}"
    fi
    
    # 检查checkpoint文件是否存在
    if [ ! -f "$FULL_CHECKPOINT_PATH" ]; then
        echo "警告: 指定的checkpoint文件不存在: $FULL_CHECKPOINT_PATH"
        echo "将尝试自动从latest_checkpoint.txt恢复"
    else
        echo "使用指定的checkpoint: $FULL_CHECKPOINT_PATH"
        TRAIN_CMD="${TRAIN_CMD} checkpoint.load_path=\"${RESUME_FROM_CHECKPOINT}\""
        
        # 设置是否加载训练状态
        if [ "$LOAD_TRAINING_STATE" = "true" ] || [ "$LOAD_TRAINING_STATE" = "True" ] || [ "$LOAD_TRAINING_STATE" = "1" ]; then
            TRAIN_CMD="${TRAIN_CMD} checkpoint.load_training_state=true"
            echo "将加载训练状态（优化器、调度器等）"
        else
            TRAIN_CMD="${TRAIN_CMD} checkpoint.load_training_state=false"
            echo "只加载模型权重，不加载训练状态"
        fi
    fi
else
    # 如果没有指定checkpoint路径，训练代码会自动检查latest_checkpoint.txt
    # 如果存在，会自动恢复训练；如果不存在，会从load_path加载预训练权重
    echo "未指定checkpoint路径，将自动检测latest_checkpoint.txt或使用预训练权重"
fi

# 根据是否使用mask通道添加相应参数
# 注意：use_mask_channel 不是 Config 类的字段，不能作为命令行参数传递
# 只能通过覆盖相关的通道数配置来实现
if [ "$USE_MASK_FLAG" = "false" ]; then
    echo "配置: 不使用mask通道（13通道）"
    TRAIN_CMD="${TRAIN_CMD} model.config.network.in_channels=13"
    TRAIN_CMD="${TRAIN_CMD} model.config.network.out_channels=13"
    TRAIN_CMD="${TRAIN_CMD} dataloader_train.dataset.target_channels=13"
    TRAIN_CMD="${TRAIN_CMD} dataloader_train.dataset.include_mask_channel=false"
    TRAIN_CMD="${TRAIN_CMD} dataloader_val.dataset.target_channels=13"
    TRAIN_CMD="${TRAIN_CMD} dataloader_val.dataset.include_mask_channel=false"
    TRAIN_CMD="${TRAIN_CMD} checkpoint.jit.input_shape=[1,13,256,256]"
else
    echo "配置: 使用mask通道（14通道）"
fi

# 打印完整命令
echo ""
echo "执行命令:"
echo "$TRAIN_CMD"
echo ""
echo "开始训练..."
echo "日志将保存到: $LOG_FILE"
echo ""

# 执行训练命令并保存日志
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

