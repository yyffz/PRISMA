# Satellite Tokenizer Training, Inference, and Evaluation

This repository contains the training, inference, and evaluation code for adapting
NVIDIA Cosmos-Predict1 tokenizers to satellite and precipitation observations.
It keeps the original Cosmos-Predict1 tokenizer stack and adds satellite-specific
datasets, model configurations, channel adapters, shell entry points, and
evaluation utilities.

The satellite tokenizer fine-tuning and evaluation protocol follows the PRISMA
study, "A plug-and-play generative framework for multi-satellite precipitation
estimation" (arXiv:2605.14426): https://arxiv.org/abs/2605.14426. In particular,
the AGRI tokenizer setup uses multi-channel FY-4B AGRI observations, optional
observation masks, and reconstruction metrics computed on valid satellite
observation regions.

## Project Overview

The codebase is organized around three main workflows:

- Satellite tokenizer fine-tuning from Cosmos-Tokenize1 checkpoints.
- Batch tokenizer inference and reconstruction evaluation for AGRI satellite
  observations.
- Supporting preprocessing and visualization scripts for radar, nowcasting, and
  satellite data products.

Key files and directories:

```text
cosmos_predict1/tokenizer_satellite/       Satellite tokenizer models, datasets, training, inference
cosmos_predict1/tokenizer/                 Original Cosmos tokenizer implementation
cosmos_predict1/autoregressive/            Cosmos autoregressive inference and training modules
cosmos_predict1/diffusion/                 Cosmos diffusion inference and training modules
run_train_agri_tokenizer.sh                AGRI video tokenizer fine-tuning entry point
run_train_agri_image_tokenizer.sh          AGRI image tokenizer fine-tuning entry point
run_train_gmi_tokenizer.sh                 GMI tokenizer fine-tuning entry point
run_train_imerg_tokenizer.sh               IMERG precipitation tokenizer fine-tuning entry point
run_train_dpr_ka_tokenizer.sh              DPR Ka tokenizer fine-tuning entry point
run_batch_evaluate_agri_tokenizer_region.sh AGRI batch inference and regional evaluation
inference_Tokenizer.sh                     Original Cosmos tokenizer inference example
datasets/                                 Local dataset mount point, not intended for Git
checkpoints/                              Local checkpoint directory, not intended for Git
outputs/                                  Evaluation and reconstruction outputs
```

## Environment Setup

The project expects Python 3.10, PyTorch with CUDA support, and the Cosmos
training dependencies. A conda environment file and pip requirements are
provided.

```bash
cd train_infer_eval_code

conda env create --file cosmos-predict1.yaml
conda activate cosmos-predict1

pip install -r requirements.txt
```

For full training, install Transformer Engine and Apex as described in
`INSTALL.md`. A typical setup is:

```bash
# Patch Transformer Engine include paths in some conda installations.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10

pip install "transformer-engine[pytorch]"

# Install Apex with CUDA extensions.
# Download or clone NVIDIA/apex first, then run from the apex source directory:
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" .
```

Validate the environment:

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py --training
```

## Checkpoints and Data

Download or place Cosmos tokenizer checkpoints under `checkpoints/`. The AGRI
video tokenizer configuration expects:

```text
checkpoints/Cosmos-Tokenize1-CV4x8x8-360p/model.pt
```

The AGRI image, GMI, IMERG, and DPR Ka configurations use image tokenizer
checkpoints such as:

```text
checkpoints/Cosmos-Tokenize1-CI8x8-360p/model.pt
```

You can download Cosmos tokenizer checkpoints with:

```bash
python -m scripts.download_tokenizer_checkpoints --checkpoint_dir checkpoints
```

Large files should stay outside Git:

```text
datasets/
checkpoints/
logs/
outputs/
*.pt
*.jit
*.npy
*.npz
```

## Data Format

The AGRI video tokenizer uses preprocessed NPZ files. The default format is:

```text
agri_data:        float32, shape [9, T, H, W] or [9, H, W], normalized to [-1, 1]
observation_mask: float32, shape [T, H, W] or [H, W], where 1 means valid observation
```

The default AGRI video setup uses `T=7` and `H=W=256`. With
`--use_mask_channel true`, the mask is appended as an additional input channel,
so the model input becomes `[10, 7, 256, 256]`. Metrics normally exclude this
mask channel and are computed only over valid observation pixels.

## Hardware Requirements

Training and evaluation require NVIDIA GPUs.

Recommended baseline:

- CUDA-capable NVIDIA GPU with recent drivers.
- CUDA 12.x compatible PyTorch environment.
- 1 x 80 GB GPU for conservative AGRI video tokenizer fine-tuning with
  `batch_size=1` and `latent_channels=32`.
- Multi-GPU training through `torchrun --nproc_per_node=<NUM_GPUS>` for larger
  batches or faster training.
- At least 32 GB system RAM, with more recommended for large NPZ datasets and
  many dataloader workers.

Lower-memory GPUs may work with smaller `batch_size`, fewer dataloader workers,
`float16`/`bfloat16`, or smaller channel/latent settings, but this should be
validated per dataset.

## Training

All satellite tokenizer training scripts call:

```bash
torchrun -m cosmos_predict1.tokenizer_satellite.training.train \
    --config=cosmos_predict1/tokenizer_satellite/training/configs/config.py \
    -- experiment=<experiment_name> ...
```

The shell scripts wrap this command and expose the common overrides.

### AGRI Video Tokenizer

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

Resume from a checkpoint:

```bash
bash run_train_agri_tokenizer.sh \
    --data_pattern 'datasets/AGRI_processed/preprocessed_video/*_agri_video_*.npz' \
    --resume_from_checkpoint checkpoints/satellite_tokenizer/agri/<run>/checkpoints/iter_000240000.pt \
    --load_training_state true
```

The AGRI video script writes logs to `logs/` and checkpoints under:

```text
checkpoints/satellite_tokenizer/agri/<run_name>/checkpoints/
```

Typical checkpoint artifacts include:

```text
iter_XXXXXXXXX.pt
iter_XXXXXXXXX_ema.jit
iter_XXXXXXXXX_enc.jit
iter_XXXXXXXXX_dec.jit
latest_checkpoint.txt
```

Use `*_ema.jit` for direct autoencoder reconstruction evaluation, or use
matching `*_enc.jit` and `*_dec.jit` files.

### Other Tokenizers

GMI:

```bash
bash run_train_gmi_tokenizer.sh \
    --data_pattern 'datasets/GMI_processed/preprocessed/*_GMI.npz' \
    --use_mask_channel true \
    --num_gpus 1 \
    --batch_size 8
```

IMERG:

```bash
bash run_train_imerg_tokenizer.sh \
    --data_pattern 'datasets/IMERG_processed/*_imerg_*.npz' \
    --use_qi_channel false \
    --num_gpus 2 \
    --batch_size 8
```

DPR Ka:

```bash
bash run_train_dpr_ka_tokenizer.sh \
    --data_pattern 'datasets/DPR_Ka_processed/*_DPR_Ka.npz' \
    --num_gpus 2 \
    --batch_size 8
```

AGRI image tokenizer:

```bash
bash run_train_agri_image_tokenizer.sh \
    --data_pattern 'datasets/AGRI_processed/preprocessed_image/*_agri_image_*.npz' \
    --use_mask_channel true \
    --num_gpus 4 \
    --batch_size 8
```

Run each script with `--help` for the full list of supported overrides.

## Inference and Evaluation

For AGRI regional reconstruction evaluation:

```bash
bash run_batch_evaluate_agri_tokenizer_region.sh \
    --data_pattern 'datasets/AGRI_processed/preprocessed_video/*_agri_video_*.npz' \
    --checkpoint checkpoints/satellite_tokenizer/agri/<run>/checkpoints/iter_000240000_ema.jit \
    --output_dir outputs/agri_tokenizer_region_eval \
    --region 0:256,0:256 \
    --max_samples 100 \
    --save_reconstructions true
```

If `--region` is omitted, the full image is evaluated. Region syntax is:

```text
row_start:row_end,col_start:col_end
```

The same evaluation can be launched directly as a Python module:

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

Evaluation outputs:

```text
agri_tokenizer_region_metrics.csv       Per-sample metrics
agri_tokenizer_region_per_channel.csv   Per-sample, per-channel metrics
agri_tokenizer_region_summary.json      Global and per-channel summary
reconstructions/*.npz                   Optional saved inputs, reconstructions, errors, masks
```

The evaluator reports MAE, MSE, RMSE, and PSNR in the normalized `[-1, 1]`
space. By default, metrics are computed only where `observation_mask=1`, and the
mask channel is not included in reconstruction metrics.

For the original Cosmos tokenizer inference example:

```bash
bash inference_Tokenizer.sh
```

## Citation

If this code is useful for your research, please cite the related satellite
tokenizer and PRISMA work:

```bibtex
@article{yang2026prisma,
  title={A plug-and-play generative framework for multi-satellite precipitation estimation},
  author={Yang, Yunfan and Sun, Haofei and Sun, Xiuyu and Han, Wei and Xu, Xiaoze and Song, Xingtao and Li, Jun and Gao, Zhiqiu and Huang, Wei and Li, Hao},
  journal={arXiv preprint arXiv:2605.14426},
  year={2026}
}
```

## Acknowledgements

This project builds on NVIDIA Cosmos-Predict1 and Cosmos-Tokenize1. We thank the
Cosmos team and contributors for releasing the world foundation model code,
tokenizer architectures, training framework, checkpoints, and documentation that
made this satellite adaptation possible.

Cosmos-Predict1 source code is released under the Apache 2.0 License. Cosmos
model weights are released under the NVIDIA Open Model License. Please review
`LICENSE`, `ATTRIBUTIONS.md`, and the original Cosmos project terms before using
or redistributing code, checkpoints, or derived models.
