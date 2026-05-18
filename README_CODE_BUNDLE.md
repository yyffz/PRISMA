# Train / Inference / Evaluation Code Bundle

This directory is a code-only bundle prepared from the current repository for
training, inference, and evaluation.

## Included

- `cosmos_predict1/`: Python package used by training, inference, and evaluation.
- Root task scripts:
  - `run_train_*_tokenizer.sh`
  - `run_batch_evaluate_agri_tokenizer_region.sh`
  - `inference_*.sh`
  - `PostTraining_Tokenizer.sh`
- Config and environment files:
  - `pyproject.toml`
  - `requirements.txt`
  - `cosmos-predict1.yaml`
  - `Dockerfile`
- Helper scripts and docs:
  - `scripts/`
  - `examples/`
  - `workspace/*.py`
  - `workspace/Tokenizer_prep/*.py`
  - project README and install docs
- Small example assets needed by the sample inference scripts.

## Not Included

The following are intentionally not copied:

- Checkpoints and model weights.
- Training logs.
- Generated outputs.
- Large datasets and case files.
- Large generated `.npy`, `.npz`, `.nc`, `.grib`, checkpoint, and output files.

## Where To Put Checkpoints

Upload or copy checkpoints into:

```text
train_infer_eval_code/checkpoints/
```

Keep the original directory structure expected by the scripts. Examples:

```text
train_infer_eval_code/checkpoints/Cosmos-Predict1-4B/
train_infer_eval_code/checkpoints/satellite_tokenizer/agri/<run>/checkpoints/
train_infer_eval_code/checkpoints/Cosmos-Tokenize1-DV4x8x8-360p/
```

## Typical Commands

Run commands from this directory:

```bash
cd train_infer_eval_code
```

Train AGRI tokenizer:

```bash
bash run_train_agri_tokenizer.sh --data_pattern '/path/to/data/*_agri_video_*.npz'
```

Evaluate AGRI tokenizer:

```bash
bash run_batch_evaluate_agri_tokenizer_region.sh \
  --data_pattern '/path/to/data/*_agri_video_*.npz' \
  --checkpoint 'checkpoints/satellite_tokenizer/agri/<run>/checkpoints/iter_xxxxxx_ema.jit'
```

Tokenizer inference:

```bash
bash inference_Tokenizer.sh
```

Autoregressive inference:

```bash
bash inference_AR_CREF.sh
```

If data is outside this directory, pass absolute paths through the script
arguments such as `--data_pattern` or `--input_image_or_video_path`.
