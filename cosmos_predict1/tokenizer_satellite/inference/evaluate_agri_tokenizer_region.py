# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch reconstruction evaluation for AGRI satellite tokenizer checkpoints.

The expected AGRI input file is a preprocessed NPZ containing:
  - agri_data: [C, T, H, W] or [C, H, W], normalized to [-1, 1].
  - observation_mask: [T, H, W] or [H, W], where 1 means valid observation.

If --use_mask_channel is enabled and agri_data has 9 channels, the observation
mask is appended as the 10th channel before inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


DEFAULT_CHANNEL_NAMES = [
    "agri_band_06",
    "agri_band_07",
    "agri_band_08",
    "agri_band_09",
    "agri_band_10",
    "agri_band_11",
    "agri_band_12",
    "agri_band_13",
    "agri_band_14",
    "observation_mask",
]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_region(region: str | None) -> tuple[slice, slice] | None:
    """Parse a region string formatted as 'row0:row1,col0:col1'."""
    if region is None or region == "":
        return None
    try:
        row_part, col_part = region.split(",")
        row_start, row_end = [int(v) for v in row_part.split(":")]
        col_start, col_end = [int(v) for v in col_part.split(":")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid region '{region}'. Expected format: row_start:row_end,col_start:col_end"
        ) from exc
    if row_start < 0 or col_start < 0 or row_end <= row_start or col_end <= col_start:
        raise argparse.ArgumentTypeError(f"Invalid region bounds: {region}")
    return slice(row_start, row_end), slice(col_start, col_end)


def get_filepaths(data_pattern: str, limit: int | None = None) -> list[str]:
    if data_pattern.startswith("@"):
        list_path = data_pattern[1:]
        with open(list_path, "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        paths = glob(data_pattern, recursive=True)
    paths = sorted({os.path.abspath(path) for path in paths if not os.path.basename(path).startswith(".")})
    if limit is not None:
        paths = paths[:limit]
    return paths


def normalize_mask(mask: np.ndarray | None, time_steps: int, height: int, width: int) -> np.ndarray | None:
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = np.broadcast_to(mask[None, :, :], (time_steps, height, width)).copy()
    elif mask.ndim == 3:
        if mask.shape[0] == time_steps:
            pass
        elif mask.shape[-1] == time_steps:
            mask = np.transpose(mask, (2, 0, 1))
        elif mask.shape[0] == 1:
            mask = np.broadcast_to(mask, (time_steps, height, width)).copy()
        else:
            mask = mask[0:1, :, :]
            mask = np.broadcast_to(mask, (time_steps, height, width)).copy()
    else:
        raise ValueError(f"Unsupported observation_mask shape: {mask.shape}")
    if mask.shape != (time_steps, height, width):
        raise ValueError(f"Mask shape {mask.shape} does not match data shape {(time_steps, height, width)}")
    return np.clip(mask, 0.0, 1.0)


def load_agri_npz(
    filepath: str,
    data_key: str,
    mask_key: str,
    use_mask_channel: bool,
    region: tuple[slice, slice] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(filepath, allow_pickle=False) as npz:
        if data_key not in npz:
            available = ", ".join(npz.files)
            raise KeyError(f"{filepath} missing '{data_key}'. Available keys: {available}")
        data = np.asarray(npz[data_key], dtype=np.float32)
        mask = np.asarray(npz[mask_key], dtype=np.float32) if mask_key in npz else None

    if data.ndim == 3:
        data = data[:, None, :, :]
    if data.ndim != 4:
        raise ValueError(f"{filepath} {data_key} should be [C,T,H,W] or [C,H,W], got {data.shape}")

    data = np.nan_to_num(data, nan=-1.0, posinf=1.0, neginf=-1.0)
    channels, time_steps, height, width = data.shape
    mask = normalize_mask(mask, time_steps, height, width)

    if use_mask_channel:
        if channels == 9:
            if mask is None:
                mask = np.ones((time_steps, height, width), dtype=np.float32)
            data = np.concatenate([data, mask[None, :, :, :]], axis=0)
        elif channels == 10:
            if mask is None:
                mask = data[-1]
        else:
            raise ValueError(f"{filepath} expected 9 or 10 AGRI channels with mask mode, got {channels}")
    else:
        if channels == 10:
            if mask is None:
                mask = data[-1]
            data = data[:9]
        elif channels != 9:
            raise ValueError(f"{filepath} expected 9 AGRI channels without mask mode, got {channels}")

    if region is not None:
        row_slice, col_slice = region
        data = data[:, :, row_slice, col_slice]
        if mask is not None:
            mask = mask[:, row_slice, col_slice]

    return data.astype(np.float32, copy=False), mask


def extract_reconstruction(output: Any, torch_module: Any) -> Any:
    if isinstance(output, torch_module.Tensor):
        return output
    if isinstance(output, dict):
        return output["reconstructions"]
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "reconstructions"):
        return output.reconstructions
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


class Autoencoder:
    def __init__(
        self,
        checkpoint: str | None,
        checkpoint_enc: str | None,
        checkpoint_dec: str | None,
        device: str,
        dtype: Any,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.model = self.torch.jit.load(checkpoint, map_location=device).eval() if checkpoint else None
        self.encoder = self.torch.jit.load(checkpoint_enc, map_location=device).eval() if checkpoint_enc else None
        self.decoder = self.torch.jit.load(checkpoint_dec, map_location=device).eval() if checkpoint_dec else None
        if self.model is None and (self.encoder is None or self.decoder is None):
            raise ValueError("Provide either --checkpoint or both --checkpoint_enc and --checkpoint_dec")

    def __call__(self, batch: Any) -> Any:
        with self.torch.inference_mode():
            batch = batch.to(device=self.device, dtype=self.dtype)
            if self.model is not None:
                return extract_reconstruction(self.model(batch), self.torch)
            latent = self.encoder(batch)
            if isinstance(latent, (tuple, list)):
                latent = latent[0]
            return extract_reconstruction(self.decoder(latent), self.torch)


def psnr_from_mse(mse: float, data_range: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def compute_metrics(
    truth: np.ndarray,
    recon: np.ndarray,
    mask: np.ndarray | None,
    data_channels: int,
    mask_metrics: bool,
    data_range: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    truth = truth[:data_channels]
    recon = recon[:data_channels]
    diff = recon - truth

    if mask_metrics and mask is not None:
        valid = (mask > 0.5).astype(np.float64)
        valid = np.broadcast_to(valid[None, :, :, :], diff.shape)
    else:
        valid = np.ones_like(diff, dtype=np.float64)

    count = float(valid.sum())
    if count <= 0:
        raise ValueError("No valid pixels for metric computation")

    abs_err = np.abs(diff) * valid
    sq_err = np.square(diff, dtype=np.float64) * valid
    mae = float(abs_err.sum() / count)
    mse = float(sq_err.sum() / count)
    rmse = float(math.sqrt(mse))
    overall = {
        "valid_count": count,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr_from_mse(mse, data_range),
    }

    per_channel = []
    for channel_idx in range(data_channels):
        channel_valid = valid[channel_idx]
        channel_count = float(channel_valid.sum())
        if channel_count <= 0:
            per_channel.append(
                {
                    "channel": channel_idx,
                    "valid_count": 0.0,
                    "mae": float("nan"),
                    "mse": float("nan"),
                    "rmse": float("nan"),
                    "psnr": float("nan"),
                }
            )
            continue
        channel_diff = diff[channel_idx]
        channel_mae = float((np.abs(channel_diff) * channel_valid).sum() / channel_count)
        channel_mse = float((np.square(channel_diff, dtype=np.float64) * channel_valid).sum() / channel_count)
        per_channel.append(
            {
                "channel": channel_idx,
                "valid_count": channel_count,
                "mae": channel_mae,
                "mse": channel_mse,
                "rmse": float(math.sqrt(channel_mse)),
                "psnr": psnr_from_mse(channel_mse, data_range),
            }
        )
    return overall, per_channel


def write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AGRI tokenizer reconstruction quality on NPZ files.")
    parser.add_argument("--data_pattern", required=True, help="Glob pattern or @filelist for AGRI NPZ inputs.")
    parser.add_argument("--checkpoint", default=None, help="Full autoencoder TorchScript checkpoint, e.g. *_ema.jit.")
    parser.add_argument("--checkpoint_enc", default=None, help="Encoder TorchScript checkpoint, e.g. *_enc.jit.")
    parser.add_argument("--checkpoint_dec", default=None, help="Decoder TorchScript checkpoint, e.g. *_dec.jit.")
    parser.add_argument("--output_dir", default="outputs/agri_tokenizer_eval", help="Directory for CSV/JSON outputs.")
    parser.add_argument("--data_key", default="agri_data", help="NPZ data key.")
    parser.add_argument("--mask_key", default="observation_mask", help="NPZ observation mask key.")
    parser.add_argument("--use_mask_channel", type=str2bool, default=True, help="Append/use the mask as model channel 10.")
    parser.add_argument("--mask_metrics", dest="mask_metrics", action="store_true", help="Compute metrics only on valid mask.")
    parser.add_argument("--no_mask_metrics", dest="mask_metrics", action="store_false", help="Compute metrics on all pixels.")
    parser.set_defaults(mask_metrics=True)
    parser.add_argument("--evaluate_mask_channel", action="store_true", help="Include mask channel in metrics.")
    parser.add_argument("--region", default=None, help="Optional crop as row_start:row_end,col_start:col_end.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit the number of evaluated samples.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Inference dtype.")
    parser.add_argument("--data_range", type=float, default=2.0, help="Value range for PSNR. Use 2.0 for [-1,1].")
    parser.add_argument("--save_reconstructions", action="store_true", help="Save per-sample reconstruction NPZ files.")
    parser.add_argument("--output_csv", default=None, help="Optional per-file metric CSV path.")
    parser.add_argument("--per_channel_csv", default=None, help="Optional per-channel metric CSV path.")
    parser.add_argument("--summary_json", default=None, help="Optional summary JSON path.")
    parser.add_argument("--channel_names", default=None, help="Comma-separated names for AGRI/mask channels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    region = parse_region(args.region)
    os.makedirs(args.output_dir, exist_ok=True)

    output_csv = args.output_csv or os.path.join(args.output_dir, "agri_tokenizer_region_metrics.csv")
    per_channel_csv = args.per_channel_csv or os.path.join(args.output_dir, "agri_tokenizer_region_per_channel.csv")
    summary_json = args.summary_json or os.path.join(args.output_dir, "agri_tokenizer_region_summary.json")
    recon_dir = os.path.join(args.output_dir, "reconstructions")
    if args.save_reconstructions:
        os.makedirs(recon_dir, exist_ok=True)

    paths = get_filepaths(args.data_pattern, args.max_samples)
    if not paths:
        raise FileNotFoundError(f"No files matched data_pattern: {args.data_pattern}")

    dtype = getattr(torch, args.dtype)
    autoencoder = Autoencoder(args.checkpoint, args.checkpoint_enc, args.checkpoint_dec, args.device, dtype)

    channel_names = args.channel_names.split(",") if args.channel_names else DEFAULT_CHANNEL_NAMES
    data_channels = 10 if args.evaluate_mask_channel and args.use_mask_channel else 9

    rows: list[dict[str, Any]] = []
    per_channel_rows: list[dict[str, Any]] = []
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0.0
    channel_abs = [0.0 for _ in range(data_channels)]
    channel_sq = [0.0 for _ in range(data_channels)]
    channel_count = [0.0 for _ in range(data_channels)]

    for filepath in tqdm(paths, desc="Evaluating AGRI tokenizer"):
        data, mask = load_agri_npz(filepath, args.data_key, args.mask_key, args.use_mask_channel, region)
        tensor = torch.from_numpy(data[None, ...])
        recon = autoencoder(tensor)
        recon_np = recon[0].float().cpu().numpy()

        if recon_np.shape != data.shape:
            raise RuntimeError(f"Reconstruction shape {recon_np.shape} does not match input shape {data.shape}")

        metrics, per_channel = compute_metrics(
            data,
            recon_np,
            mask,
            data_channels=data_channels,
            mask_metrics=args.mask_metrics,
            data_range=args.data_range,
        )
        row = {
            "file": filepath,
            "base_name": os.path.basename(filepath),
            "channels": int(data.shape[0]),
            "time_steps": int(data.shape[1]),
            "height": int(data.shape[2]),
            "width": int(data.shape[3]),
            "region": args.region or "full",
            "valid_count": metrics["valid_count"],
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "psnr": metrics["psnr"],
        }
        rows.append(row)

        total_count += metrics["valid_count"]
        total_abs += metrics["mae"] * metrics["valid_count"]
        total_sq += metrics["mse"] * metrics["valid_count"]

        for item in per_channel:
            c = int(item["channel"])
            item_row = {
                "file": filepath,
                "base_name": os.path.basename(filepath),
                "channel": c,
                "channel_name": channel_names[c] if c < len(channel_names) else f"channel_{c}",
                "region": args.region or "full",
                "valid_count": item["valid_count"],
                "mae": item["mae"],
                "mse": item["mse"],
                "rmse": item["rmse"],
                "psnr": item["psnr"],
            }
            per_channel_rows.append(item_row)
            if item["valid_count"] > 0:
                channel_count[c] += item["valid_count"]
                channel_abs[c] += item["mae"] * item["valid_count"]
                channel_sq[c] += item["mse"] * item["valid_count"]

        if args.save_reconstructions:
            stem = Path(filepath).stem
            save_path = os.path.join(recon_dir, f"{stem}_reconstruction.npz")
            payload = {
                "input": data.astype(np.float32),
                "reconstruction": recon_np.astype(np.float32),
                "error": (recon_np - data).astype(np.float32),
            }
            if mask is not None:
                payload["observation_mask"] = mask.astype(np.float32)
            np.savez_compressed(save_path, **payload)

    if total_count <= 0:
        raise RuntimeError("No valid pixels were evaluated.")

    summary_channels = []
    for c in range(data_channels):
        if channel_count[c] <= 0:
            summary_channels.append(
                {
                    "channel": c,
                    "channel_name": channel_names[c] if c < len(channel_names) else f"channel_{c}",
                    "valid_count": 0,
                    "mae": float("nan"),
                    "mse": float("nan"),
                    "rmse": float("nan"),
                    "psnr": float("nan"),
                }
            )
            continue
        channel_mse = channel_sq[c] / channel_count[c]
        summary_channels.append(
            {
                "channel": c,
                "channel_name": channel_names[c] if c < len(channel_names) else f"channel_{c}",
                "valid_count": channel_count[c],
                "mae": channel_abs[c] / channel_count[c],
                "mse": channel_mse,
                "rmse": math.sqrt(channel_mse),
                "psnr": psnr_from_mse(channel_mse, args.data_range),
            }
        )

    summary_mse = total_sq / total_count
    summary = {
        "num_files": len(rows),
        "data_pattern": args.data_pattern,
        "checkpoint": args.checkpoint,
        "checkpoint_enc": args.checkpoint_enc,
        "checkpoint_dec": args.checkpoint_dec,
        "region": args.region or "full",
        "mask_metrics": args.mask_metrics,
        "use_mask_channel": args.use_mask_channel,
        "evaluate_mask_channel": args.evaluate_mask_channel,
        "valid_count": total_count,
        "mae": total_abs / total_count,
        "mse": summary_mse,
        "rmse": math.sqrt(summary_mse),
        "psnr": psnr_from_mse(summary_mse, args.data_range),
        "mean_file_mae": finite_mean([row["mae"] for row in rows]),
        "mean_file_rmse": finite_mean([row["rmse"] for row in rows]),
        "mean_file_psnr": finite_mean([row["psnr"] for row in rows]),
        "channels": summary_channels,
    }

    write_csv(
        output_csv,
        rows,
        ["file", "base_name", "channels", "time_steps", "height", "width", "region", "valid_count", "mae", "mse", "rmse", "psnr"],
    )
    write_csv(
        per_channel_csv,
        per_channel_rows,
        ["file", "base_name", "channel", "channel_name", "region", "valid_count", "mae", "mse", "rmse", "psnr"],
    )
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved per-file metrics: {output_csv}")
    print(f"Saved per-channel metrics: {per_channel_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
