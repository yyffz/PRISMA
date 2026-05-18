# Cosmos Tokenizer 卫星 AGRI 适配说明

本文档整理了基于 NVIDIA Cosmos-Predict1 视觉 Tokenizer 的卫星观测适配流程，重点覆盖 AGRI 多通道卫星观测的环境配置、训练、推理与区域评估。

## 工作概览

- 基座模型：`Cosmos-Tokenize1-CV4x8x8-360p` 连续视频 tokenizer。
- 任务数据：预处理后的 AGRI 卫星观测 NPZ，默认 `agri_data` 为 `[9, 7, 256, 256]`，9 个中红外/热红外通道。
- Mask 通道：默认把 `observation_mask` 追加为第 10 个通道，输入形状为 `[10, 7, 256, 256]`。
- 结构改动：复用 Cosmos 因子化视频 tokenizer，将输入/输出通道扩展到 9 或 10，`latent_channels=z_channels=32`。
- 初始化策略：`channel_init_strategy=all_pretrained`，从 RGB 预训练权重循环复制初始化新增通道。
- 损失配置：以 L1 重建为主，KL 低权重正则；禁用自然图像 LPIPS/Gram/Correlation；5k step 后启用 video consistency。

## 关键文件

- 环境：`cosmos-predict1.yaml`、`requirements.txt`、`INSTALL.md`
- 训练入口：`run_train_agri_tokenizer.sh`
- 训练配置：`cosmos_predict1/tokenizer_satellite/training/configs/experiments/agri_training.py`
- 数据加载：`cosmos_predict1/tokenizer_satellite/training/datasets/gmi_dataset.py`
- 批量区域评估：`run_batch_evaluate_agri_tokenizer_region.sh`
- 评估实现：`cosmos_predict1/tokenizer_satellite/inference/evaluate_agri_tokenizer_region.py`

## 环境配置

```bash
conda env create --file cosmos-predict1.yaml
conda activate cosmos-predict1
pip install -r requirements.txt

# Transformer Engine / Apex 按 INSTALL.md 中说明安装。
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py --training
```

训练前需要准备 Cosmos tokenizer 预训练权重：

```bash
python3 -m scripts.download_tokenizer_checkpoints --checkpoint_dir checkpoints
```

AGRI 配置默认读取：

```text
checkpoints/Cosmos-Tokenize1-CV4x8x8-360p/model.pt
```

如果机器不能联网，请手动下载并放到上述目录；同时确认 `agri_training.py` 中 `vgg_weights_path` 指向本机可读的 `vgg16-397923af.pth`，虽然当前 LPIPS 权重为 0，但初始化时仍可能访问该路径。

## 数据格式

默认使用预处理后的 NPZ：

```text
agri_data:        float32, [9, T, H, W] 或 [9, H, W]，数值范围 [-1, 1]
observation_mask: float32, [T, H, W] 或 [H, W]，1=有效观测，0=无效观测
```

视频训练默认 `T=7, H=W=256`。如果 `agri_data` 已经包含 mask 通道，可使用 `[10, T, H, W]`；评估脚本会在 `--use_mask_channel true` 时识别 9/10 通道两种情况。

## 训练

单卡示例：

```bash
bash run_train_agri_tokenizer.sh \
    --data_pattern 'datasets/AGRI_processed/preprocessed_video/*_agri_video_*.npz' \
    --use_mask_channel true \
    --num_gpus 1 \
    --cuda_visible_devices 0 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --warmup_steps 10000 \
    --latent_channels 32
```

断点续训：

```bash
bash run_train_agri_tokenizer.sh \
    --data_pattern 'datasets/AGRI_processed/preprocessed_video/*_agri_video_*.npz' \
    --resume_from_checkpoint checkpoints/satellite_tokenizer/agri/<run>/checkpoints/iter_000240000.pt \
    --load_training_state true
```

训练输出位于：

```text
checkpoints/satellite_tokenizer/agri/<run_name>/checkpoints/
```

每个保存点通常包含：

```text
iter_XXXXXXXXX.pt
iter_XXXXXXXXX_ema.jit
iter_XXXXXXXXX_enc.jit
iter_XXXXXXXXX_dec.jit
latest_checkpoint.txt
```

其中 `*_ema.jit` 可直接用于 autoencoder 重建评估；也可以组合 `*_enc.jit` 与 `*_dec.jit`。

## 推理与区域评估

批量评估脚本会读取 AGRI NPZ，执行 tokenizer 重建，并输出归一化空间 `[-1, 1]` 上的 MAE、MSE、RMSE、PSNR。默认只在 `observation_mask=1` 的有效区域统计指标，且不把 mask 通道计入指标。

```bash
bash run_batch_evaluate_agri_tokenizer_region.sh \
    --data_pattern 'datasets/AGRI_processed/preprocessed_video/*_agri_video_*.npz' \
    --checkpoint checkpoints/satellite_tokenizer/agri/<run>/checkpoints/iter_000240000_ema.jit \
    --output_dir outputs/agri_tokenizer_region_eval \
    --region 0:256,0:256 \
    --max_samples 100 \
    --save_reconstructions true
```

不指定 `--region` 时评估全图。区域格式为：

```text
row_start:row_end,col_start:col_end
```

输出文件：

- `agri_tokenizer_region_metrics.csv`：逐样本整体指标。
- `agri_tokenizer_region_per_channel.csv`：逐样本逐通道指标。
- `agri_tokenizer_region_summary.json`：全局汇总与逐通道汇总。
- `reconstructions/*.npz`：当 `--save_reconstructions true` 时保存输入、重建、误差和 mask。

也可以直接调用 Python 模块：

```bash
PYTHONPATH=$(pwd) python -m cosmos_predict1.tokenizer_satellite.inference.evaluate_agri_tokenizer_region \
    --data_pattern 'datasets/AGRI_processed/preprocessed_video/*_agri_video_*.npz' \
    --checkpoint checkpoints/satellite_tokenizer/agri/<run>/checkpoints/iter_000240000_ema.jit \
    --output_dir outputs/agri_tokenizer_region_eval \
    --use_mask_channel true \
    --mask_metrics \
    --device cuda \
    --dtype bfloat16
```

## GitHub 发布建议

- 不要提交 `datasets/`、`checkpoints/`、`logs/`、`outputs/`、`*.pt`、`*.jit`、`*.npy` 等大文件。
- 在 release 或 README 中说明预训练权重、AGRI 数据和微调 checkpoint 的获取方式。
- 保留原始 Cosmos 的 Apache-2.0 代码许可与 NVIDIA Open Model License 说明。
- 对外示例尽量使用相对路径，避免提交本机 `/public/...` 或 `/cpfs...` 路径。
