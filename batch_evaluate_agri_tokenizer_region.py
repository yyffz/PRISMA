#!/usr/bin/env python3
"""
在新裁剪区域上批量评估 AGRI 视频 Tokenizer（微调 vs 预训练）

裁剪区域与 prepare_agri_video_tokens 一致：
- h=[300:1500], w=[2200:3400] -> 1200×1200
- 5 个时刻：参考时间 ref 的 ref-45min, ref-30min, ref-15min, ref, ref+15min
- 数据源：AGRI_processed 根目录下的原始 NPZ（YYYYMMDDHHMM_agri.npz）

指标说明：
- 默认仅在有效观测点（mask=1）上计算指标。无效区输入时被填为 -1.0，若在无效区也算 MSE，
  微调模型常因在无效区“重建出非 -1.0”而全图 MSE 变差，但有效区重建通常更好。
- 使用 --report_both_metrics 可同时输出「仅有效区」与「全图」指标便于对比。

异常样本过滤：
- 5 帧中若有“有效观测帧数”不足的样本（如多帧全 NaN、仅 1 帧有数据），微调解码易越界导致 MSE 爆炸，
  此类样本不纳入评估（与缺 slot 一样返回 None）。可通过 --min_valid_frames、--min_valid_ratio 调节。

用法:
    python batch_evaluate_agri_tokenizer_region.py \
        --start 2024-07-01 --end 2024-07-02 \
        --input_dir /public/share/users/sunhaofei/yyf_data/AGRI_processed \
        --pretrained_dir ... --finetuned_dir ... --finetuned_iter 210000 \
        --output_dir outputs/evaluation_results_agri_region
"""

import os
import re
import sys
import argparse
import datetime as dt
from glob import glob
from typing import List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
import json
from skimage.metrics import structural_similarity as ssim_func
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

cosmos_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cosmos_root)

from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer

# 裁剪区域（与 prepare_agri_video_tokens 一致）
H0, H1 = 300, 1500
W0, W1 = 2200, 3400
# H0, H1 = 700, 1100
# W0, W1 = 2600, 3000
# H0, H1 = 700, 1100
# W0, W1 = 2600, 3000
REGION_H = H1 - H0
REGION_W = W1 - W0

# 5 个 AGRI 时刻相对参考时间的偏移（分钟）
AGRI_VIDEO_OFFSETS_MINUTES = [-45, -30, -15, 0, 15]
NUM_AGRI_VIDEO_FRAMES = len(AGRI_VIDEO_OFFSETS_MINUTES)

# 通道与模型配置（与 batch_evaluate_agri_tokenizer 一致）
AGRI_CHANNEL_NAMES = [
    'CH07', 'CH08', 'CH09', 'CH10', 'CH11', 'CH12', 'CH13', 'CH14', 'CH15'
]
PRETRAINED_CHANNELS = [2, 4, 6]  # CH09, CH11, CH13
NUM_AGRI_CHANNELS = 9
STATS_PATH = "/public/home/sunhaofei/yyf/DGPR/channel_stats_agri_15ch.pth"
AGRI_LATENT16_FINETUNED_DIR = (
    "/public/home/sunhaofei/cosmos-predict1/checkpoints/satellite_tokenizer/agri/"
    "agri_training_2026-02-08_13-37-55/checkpoints"
)
AGRI_LATENT16_ITER = 210000
AGRI_LATENT32_FINETUNED_DIR = (
    "/public/home/sunhaofei/cosmos-predict1/checkpoints/satellite_tokenizer/agri/"
    "agri_training_2026-05-13_15-35-29/checkpoints"
)
AGRI_LATENT32_ITER = 210000
AGRI_LATENT32_OUTPUT_DIR = "outputs/evaluation_results_agri_region_latent32"
AGRI_LATENT_COMPARE_OUTPUT_DIR = "outputs/evaluation_results_agri_region_latent16_vs_latent32"

# 原始 NPZ 文件名：12 位 YYYYMMDDHHMM_agri.npz
AGRI_RAW_NPZ_RE = re.compile(r"^(\d{12})_agri\.npz$")


def slot_time_to_npz_basename(slot_dt: dt.datetime) -> str:
    """某一时刻对应的 NPZ 文件名（无路径）。"""
    return f"{slot_dt.strftime('%Y%m%d%H%M')}_agri.npz"


def build_slot_times(ref_dt: dt.datetime) -> List[dt.datetime]:
    """给定参考时间 ref_dt，返回 5 个时刻：ref-45min, ref-30min, ref-15min, ref, ref+15min。"""
    return [ref_dt + dt.timedelta(minutes=o) for o in AGRI_VIDEO_OFFSETS_MINUTES]


def iter_reference_times(
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    step_minutes: int = 15,
):
    """生成参考时间序列，使 5 个时刻均在 [start_dt, end_dt] 内。"""
    ref_start = start_dt + dt.timedelta(minutes=45)
    ref_end = end_dt - dt.timedelta(minutes=15)
    if ref_end < ref_start:
        return
    t = ref_start
    while t <= ref_end:
        yield t
        t += dt.timedelta(minutes=step_minutes)


def load_channel_stats(stats_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载通道统计量（全 15 通道；9 通道用 min_vals[6:15]）。"""
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"统计量文件不存在: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    min_vals = stats["min"].numpy() if isinstance(stats["min"], torch.Tensor) else np.array(stats["min"])
    max_vals = stats["max"].numpy() if isinstance(stats["max"], torch.Tensor) else np.array(stats["max"])
    return min_vals, max_vals


def create_observation_mask(data: np.ndarray) -> np.ndarray:
    """观测 mask：所有通道均有效（非 nan）则为 1。data: (C, H, W) -> (H, W)。"""
    return (~np.isnan(data)).all(axis=0).astype(np.float32)


def normalize_agri_data(
    data: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """归一化到 [-1, 1]。data: (C, H, W)。"""
    C = data.shape[0]
    min_v = min_vals[:C].reshape(C, 1, 1)
    max_v = max_vals[:C].reshape(C, 1, 1)
    normalized = (data - min_v) / (max_v - min_v + 1e-8) * 2.0 - 1.0
    # 检查是否在 [-1, 1] 范围内
    out_lo = int(np.sum(normalized < -1.0))
    out_hi = int(np.sum(normalized > 1.0))
    if out_lo > 0 or out_hi > 0:
        n_total = normalized.size
        print(
            f"[WARN] 归一化后数据超出 [-1, 1]: <-1 共 {out_lo} ({100*out_lo/n_total:.4f}%), "
            f">1 共 {out_hi} ({100*out_hi/n_total:.4f}%)"
        )
    normalized = np.nan_to_num(normalized, nan=-1.0)
    if mask is not None:
        normalized = np.where(mask[np.newaxis, :, :] > 0.5, normalized, -1.0)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def load_one_npz_crop_and_normalize(
    npz_path: str,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    加载单个原始 AGRI NPZ，取 CH07–CH15，归一化后裁剪 (H0:H1, W0:W1)。
    返回 (agri_region [9, REGION_H, REGION_W], mask_region [REGION_H, REGION_W])，失败返回 (None, None)。
    """
    if not os.path.isfile(npz_path):
        return None, None
    try:
        with np.load(npz_path) as f:
            if "data" not in f:
                return None, None
            raw = f["data"][6:15, ...]  # (9, H, W)
        H, W = raw.shape[1], raw.shape[2]
        if H < H1 or W < W1:
            return None, None
        obs_mask = create_observation_mask(raw)
        min_9 = min_vals[6:15]
        max_9 = max_vals[6:15]
        normalized = normalize_agri_data(raw, min_9, max_9, mask=obs_mask)
        agri_region = normalized[:, H0:H1, W0:W1]
        mask_region = obs_mask[H0:H1, W0:W1]
        agri_region = np.nan_to_num(agri_region, nan=-1.0)
        return agri_region, mask_region
    except Exception:
        return None, None


def denormalize_agri(
    data: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    channel_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """反归一化：[-1, 1] -> 物理量。"""
    data = np.clip(data, -1.0, 1.0)
    min_9 = min_vals[6:15]
    max_9 = max_vals[6:15]
    C = data.shape[0]
    if channel_indices is not None:
        min_v = min_9[channel_indices].reshape(-1, 1, 1, 1) if data.ndim == 4 else min_9[channel_indices].reshape(-1, 1, 1)
        max_v = max_9[channel_indices].reshape(-1, 1, 1, 1) if data.ndim == 4 else max_9[channel_indices].reshape(-1, 1, 1)
    else:
        min_v = min_9[:C].reshape(-1, 1, 1, 1) if data.ndim == 4 else min_9[:C].reshape(-1, 1, 1)
        max_v = max_9[:C].reshape(-1, 1, 1, 1) if data.ndim == 4 else max_9[:C].reshape(-1, 1, 1)
    return (data.astype(np.float64) + 1.0) / 2.0 * (max_v - min_v) + min_v


def calculate_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """
    计算重建指标；形状不一致时尝试对齐时间维后比较。
    mask: 可选，(T, H, W) 或 (H, W)，1=有效、0=无效；无效位置不参与计算（视为 nan）。
    """
    if original.shape != reconstructed.shape:
        if len(original.shape) == 4 and len(reconstructed.shape) == 4:
            C1, T1, H1, W1 = original.shape
            C2, T2, H2, W2 = reconstructed.shape
            if C1 == C2 and H1 == H2 and W1 == W2:
                T_min = min(T1, T2)
                original = original[:, :T_min, :, :].copy()
                reconstructed = reconstructed[:, :T_min, :, :].copy()
                if mask is not None and mask.shape[0] > T_min:
                    mask = mask[:T_min, :, :]
            else:
                return None
        else:
            return None
    else:
        original = original.copy()
        reconstructed = reconstructed.copy()

    if mask is not None:
        # 无效位置设为 nan，不参与指标计算
        if mask.ndim == 2 and original.ndim == 4:
            mask_bc = mask[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        elif mask.ndim == 3 and original.ndim == 4:
            mask_bc = mask[np.newaxis, :, :, :]  # (1, T, H, W)
        elif mask.ndim == 2 and original.ndim == 3:
            mask_bc = mask[np.newaxis, :, :]  # (1, H, W)
        else:
            mask_bc = np.broadcast_to(mask, original.shape)
        valid = mask_bc > 0.5
        original = np.where(valid, original, np.nan)
        reconstructed = np.where(valid, reconstructed, np.nan)

    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    valid_flat = ~(np.isnan(orig_flat) | np.isnan(recon_flat))
    n_valid = np.sum(valid_flat)
    if n_valid < 2:
        return None
    o = orig_flat[valid_flat]
    r = recon_flat[valid_flat]
    mse = float(np.mean((o - r) ** 2))
    mae = float(np.mean(np.abs(o - r)))
    bias = float(np.mean(r - o))
    rmse = np.sqrt(mse)
    if np.nanstd(o) > 1e-8 and np.nanstd(r) > 1e-8:
        corr = float(np.corrcoef(o, r)[0, 1])
    else:
        corr = 0.0
    data_range = float(o.max() - o.min())
    psnr = 10 * np.log10((data_range ** 2) / mse) if (data_range > 1e-8 and mse > 1e-10) else (100.0 if mse <= 1e-10 else 0.0)
    # SSIM：在空间维度上计算，用有效像素的均值填充无效区域以减少边界伪影
    orig_2d = original.reshape(-1, original.shape[-2], original.shape[-1])
    recon_2d = reconstructed.reshape(-1, reconstructed.shape[-2], reconstructed.shape[-1])
    ssim_vals = []
    for i in range(orig_2d.shape[0]):
        valid_2d = ~np.isnan(orig_2d[i])
        if valid_2d.sum() < 2:
            continue
        o_mean = float(np.nanmean(orig_2d[i]))
        r_mean = float(np.nanmean(recon_2d[i]))
        o2d = np.where(valid_2d, orig_2d[i], o_mean)
        r2d = np.where(valid_2d, recon_2d[i], r_mean)
        dr = float(o2d[valid_2d].max() - o2d[valid_2d].min())
        if dr > 1e-8:
            ssim_vals.append(ssim_func(o2d, r2d, data_range=dr))
    ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    return {
        "mse": float(mse), "mae": float(mae), "rmse": float(rmse),
        "bias": float(bias), "correlation": float(corr),
        "psnr": float(psnr), "ssim": float(ssim),
    }


class VideoTokenizerWrapper:
    """视频 Tokenizer 封装（编码+解码）。"""

    def __init__(self, checkpoint_dir: str, iter_num: Optional[int] = None, device: str = "cuda", dtype: str = "bfloat16"):
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if iter_num is not None:
            for enc_suf, dec_suf in [("09d", "09d"), ("07d", "07d")]:
                enc_path = os.path.join(checkpoint_dir, f"iter_{iter_num:{enc_suf}}_enc.jit")
                dec_path = os.path.join(checkpoint_dir, f"iter_{iter_num:{dec_suf}}_dec.jit")
                if os.path.isfile(enc_path) and os.path.isfile(dec_path):
                    self.tokenizer = CausalVideoTokenizer(checkpoint_enc=enc_path, checkpoint_dec=dec_path)
                    return
            raise FileNotFoundError(f"Encoder/Decoder not found in {checkpoint_dir} for iter={iter_num}")
        enc_path = os.path.join(checkpoint_dir, "encoder.jit")
        dec_path = os.path.join(checkpoint_dir, "decoder.jit")
        if not os.path.isfile(enc_path) or not os.path.isfile(dec_path):
            raise FileNotFoundError(f"Pretrained encoder/decoder not found: {checkpoint_dir}")
        self.tokenizer = CausalVideoTokenizer(checkpoint_enc=enc_path, checkpoint_dec=dec_path)

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(self.device).to(self.dtype)
            latent = self.tokenizer.encode(x)
            if isinstance(latent, tuple):
                latent = latent[0]
            reconstructed = self.tokenizer.decode(latent)
        return reconstructed.float().cpu()


def process_one_reference_time(
    ref_dt: dt.datetime,
    input_dir: str,
    pretrained_model: VideoTokenizerWrapper,
    finetuned_model: VideoTokenizerWrapper,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    use_mask_channel: bool = True,
    return_arrays_for_plot: bool = False,
    report_both_metrics: bool = False,
    min_valid_frames: int = 5,
    min_valid_ratio_per_frame: float = 0.5,
) -> Optional[dict]:
    """
    对一个参考时间：加载 5 个 NPZ、裁剪区域、堆叠 (9,5,H,W)，
    分别用预训练(3ch)和微调(9ch+mask)编解码，计算指标。
    若 5 帧中有效观测帧数不足（少于 min_valid_frames 帧的 valid_ratio >= min_valid_ratio_per_frame），
    视为异常样本，不纳入评估（返回 None）。
    若 return_arrays_for_plot=True 且两模型均成功，则在 result 中附加 _plot_arrays 供绘图。
    report_both_metrics: 若为 True，除默认的「仅有效区」指标外，再计算「全图」指标（*_all）。
    """
    recon_pretrained = None
    recon_finetuned_agri = None
    slot_times = build_slot_times(ref_dt)
    agri_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []

    for slot_dt in slot_times:
        npz_basename = slot_time_to_npz_basename(slot_dt)
        npz_path = os.path.join(input_dir, npz_basename)
        agri_region, mask_region = load_one_npz_crop_and_normalize(npz_path, min_vals, max_vals)
        if agri_region is None or mask_region is None:
            return None
        agri_list.append(agri_region)
        mask_list.append(mask_region)

    # (9, 5, REGION_H, REGION_W), (5, REGION_H, REGION_W)
    agri_video = np.stack(agri_list, axis=1).astype(np.float32)
    mask_video = np.stack(mask_list, axis=0).astype(np.float32)
    C, T, H, W = agri_video.shape

    # 异常样本过滤：5 帧中至少 min_valid_frames 帧的有效比例 >= min_valid_ratio_per_frame，否则不纳入评估
    valid_ratio_per_frame = [float(np.mean(mask_video[t])) for t in range(mask_video.shape[0])]
    n_valid_frames = sum(1 for r in valid_ratio_per_frame if r >= min_valid_ratio_per_frame)
    if n_valid_frames < min_valid_frames:
        return None

    result = {
        "ref_time": ref_dt.strftime("%Y%m%d%H%M%S"),
        "shape": [C, T, H, W],
    }

    input_tensor = torch.from_numpy(agri_video).float().unsqueeze(0)  # (1, 9, T, H, W)

    # 微调模型：9ch + mask -> 10ch
    try:
        if use_mask_channel:
            mask_5 = mask_video  # (5, H, W)
            mask_tensor = torch.from_numpy(mask_5).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
            input_with_mask = torch.cat([input_tensor, mask_tensor], dim=1)  # (1, 10, T, H, W)
            recon_finetuned = finetuned_model.encode_decode(input_with_mask)
        else:
            recon_finetuned = finetuned_model.encode_decode(input_tensor)
        recon_finetuned = recon_finetuned.squeeze(0).numpy()
        recon_finetuned_agri = recon_finetuned[:NUM_AGRI_CHANNELS, :, :, :]

        if return_arrays_for_plot:
            result["_recon_finetuned_agri"] = recon_finetuned_agri

        if min_vals is not None and max_vals is not None:
            agri_denorm = denormalize_agri(agri_video, min_vals, max_vals)
            recon_denorm = denormalize_agri(recon_finetuned_agri, min_vals, max_vals)
            result["finetuned"] = calculate_metrics(agri_denorm, recon_denorm, mask=mask_video)
            if report_both_metrics:
                result["finetuned_all"] = calculate_metrics(agri_denorm, recon_denorm, mask=None)
            # 逐通道指标 (9ch)
            result["finetuned_per_channel"] = {
                AGRI_CHANNEL_NAMES[ci]: calculate_metrics(
                    agri_denorm[ci:ci+1], recon_denorm[ci:ci+1], mask=mask_video
                ) for ci in range(NUM_AGRI_CHANNELS)
            }
            # 逐帧指标 (9ch)
            result["finetuned_per_frame"] = {
                f"t{ti}_offset{AGRI_VIDEO_OFFSETS_MINUTES[ti]:+d}min": calculate_metrics(
                    agri_denorm[:, ti:ti+1], recon_denorm[:, ti:ti+1], mask=mask_video[ti]
                ) for ti in range(T)
            }
        else:
            result["finetuned"] = calculate_metrics(agri_video, recon_finetuned_agri, mask=mask_video)
            if report_both_metrics:
                result["finetuned_all"] = calculate_metrics(agri_video, recon_finetuned_agri, mask=None)
            result["finetuned_per_channel"] = {
                AGRI_CHANNEL_NAMES[ci]: calculate_metrics(
                    agri_video[ci:ci+1], recon_finetuned_agri[ci:ci+1], mask=mask_video
                ) for ci in range(NUM_AGRI_CHANNELS)
            }
            result["finetuned_per_frame"] = {
                f"t{ti}_offset{AGRI_VIDEO_OFFSETS_MINUTES[ti]:+d}min": calculate_metrics(
                    agri_video[:, ti:ti+1], recon_finetuned_agri[:, ti:ti+1], mask=mask_video[ti]
                ) for ti in range(T)
            }

        # 3 通道子集（与预训练可比）
        finetuned_3ch = recon_finetuned_agri[PRETRAINED_CHANNELS, :, :, :]
        original_3ch = agri_video[PRETRAINED_CHANNELS, :, :, :]
        if min_vals is not None and max_vals is not None:
            idx = np.array(PRETRAINED_CHANNELS)
            orig_3ch_d = denormalize_agri(original_3ch, min_vals, max_vals, channel_indices=idx)
            fine_3ch_d = denormalize_agri(finetuned_3ch, min_vals, max_vals, channel_indices=idx)
            result["finetuned_3ch"] = calculate_metrics(orig_3ch_d, fine_3ch_d, mask=mask_video)
            if report_both_metrics:
                result["finetuned_3ch_all"] = calculate_metrics(orig_3ch_d, fine_3ch_d, mask=None)
        else:
            result["finetuned_3ch"] = calculate_metrics(original_3ch, finetuned_3ch, mask=mask_video)
            if report_both_metrics:
                result["finetuned_3ch_all"] = calculate_metrics(original_3ch, finetuned_3ch, mask=None)
    except Exception as e:
        result["finetuned"] = None
        result["finetuned_3ch"] = None
        print(f"[WARN] 微调模型 {ref_dt}: {e}")

    # 预训练模型：仅 3 通道
    try:
        input_3ch = input_tensor[:, PRETRAINED_CHANNELS, :, :, :]  # (1, 3, T, H, W)
        recon_pretrained = pretrained_model.encode_decode(input_3ch)
        recon_pretrained = recon_pretrained.squeeze(0).numpy()
        if return_arrays_for_plot:
            result["_recon_pretrained"] = recon_pretrained
        original_3ch = agri_video[PRETRAINED_CHANNELS, :, :, :]
        if min_vals is not None and max_vals is not None:
            idx = np.array(PRETRAINED_CHANNELS)
            orig_3ch_d = denormalize_agri(original_3ch, min_vals, max_vals, channel_indices=idx)
            recon_3ch_d = denormalize_agri(recon_pretrained, min_vals, max_vals, channel_indices=idx)
            result["pretrained"] = calculate_metrics(orig_3ch_d, recon_3ch_d, mask=mask_video)
            if report_both_metrics:
                result["pretrained_all"] = calculate_metrics(orig_3ch_d, recon_3ch_d, mask=None)
            # 逐通道指标 (3ch)
            result["pretrained_per_channel"] = {
                AGRI_CHANNEL_NAMES[PRETRAINED_CHANNELS[ci]]: calculate_metrics(
                    orig_3ch_d[ci:ci+1], recon_3ch_d[ci:ci+1], mask=mask_video
                ) for ci in range(len(PRETRAINED_CHANNELS))
            }
            # 逐帧指标 (3ch)
            result["pretrained_per_frame"] = {
                f"t{ti}_offset{AGRI_VIDEO_OFFSETS_MINUTES[ti]:+d}min": calculate_metrics(
                    orig_3ch_d[:, ti:ti+1], recon_3ch_d[:, ti:ti+1], mask=mask_video[ti]
                ) for ti in range(T)
            }
        else:
            result["pretrained"] = calculate_metrics(original_3ch, recon_pretrained, mask=mask_video)
            if report_both_metrics:
                result["pretrained_all"] = calculate_metrics(original_3ch, recon_pretrained, mask=None)
            result["pretrained_per_channel"] = {
                AGRI_CHANNEL_NAMES[PRETRAINED_CHANNELS[ci]]: calculate_metrics(
                    original_3ch[ci:ci+1], recon_pretrained[ci:ci+1], mask=mask_video
                ) for ci in range(len(PRETRAINED_CHANNELS))
            }
            result["pretrained_per_frame"] = {
                f"t{ti}_offset{AGRI_VIDEO_OFFSETS_MINUTES[ti]:+d}min": calculate_metrics(
                    original_3ch[:, ti:ti+1], recon_pretrained[:, ti:ti+1], mask=mask_video[ti]
                ) for ti in range(T)
            }
    except Exception as e:
        result["pretrained"] = None
        print(f"[WARN] 预训练模型 {ref_dt}: {e}")

    # 可选：返回绘图用数组（3 通道 Original / Pretrained / Finetuned，每帧一图）
    if return_arrays_for_plot and result.get("pretrained") and result.get("finetuned_3ch"):
        rp = result.pop("_recon_pretrained", None)
        rfa = result.pop("_recon_finetuned_agri", None)
        if rp is not None and rfa is not None:
            idx = np.array(PRETRAINED_CHANNELS)
            if min_vals is not None and max_vals is not None:
                orig_plot = denormalize_agri(agri_video[PRETRAINED_CHANNELS, :, :, :], min_vals, max_vals, channel_indices=idx)
                pre_plot = denormalize_agri(rp, min_vals, max_vals, channel_indices=idx)
                fine_plot = denormalize_agri(rfa[PRETRAINED_CHANNELS, :, :, :], min_vals, max_vals, channel_indices=idx)
                orig_9ch_plot = denormalize_agri(agri_video, min_vals, max_vals)
                fine_9ch_plot = denormalize_agri(rfa, min_vals, max_vals)
            else:
                orig_plot = agri_video[PRETRAINED_CHANNELS, :, :, :].copy()
                pre_plot = rp.copy()
                fine_plot = rfa[PRETRAINED_CHANNELS, :, :, :].copy()
                orig_9ch_plot = agri_video.copy()
                fine_9ch_plot = rfa.copy()
            result["_plot_arrays"] = (orig_plot, pre_plot, fine_plot, mask_video)
            result["_plot_9ch_arrays"] = (orig_9ch_plot, fine_9ch_plot, mask_video)
    if return_arrays_for_plot:
        result.pop("_recon_pretrained", None)
        result.pop("_recon_finetuned_agri", None)

    return result


def plot_3ch_comparison(original_3ch, recon_pretrained, recon_finetuned_3ch,
                        time_idx, output_path, title_prefix="", mask: Optional[np.ndarray] = None):
    """
    绘制微调前后重建效果对比图（CH09, CH11, CH13）
    original_3ch / recon_pretrained / recon_finetuned_3ch: (3, H, W)
    mask: 可选 (H, W)，1=有效、0=无效；仅展示有效位置，无效处显示为浅灰。
    """
    channel_names = ["CH09", "CH11", "CH13"]
    if mask is not None:
        valid = mask > 0.5
        orig_show = np.where(valid, original_3ch, np.nan)
        pre_show = np.where(valid, recon_pretrained, np.nan)
        fine_show = np.where(valid, recon_finetuned_3ch, np.nan)
    else:
        orig_show = original_3ch
        pre_show = recon_pretrained
        fine_show = recon_finetuned_3ch

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    plt.rcParams.update({"font.size": 12, "font.weight": "bold"})
    cmap_jet = plt.cm.jet.copy()
    cmap_jet.set_bad(color="lightgray", alpha=0.8)
    cmap_rdbu = plt.cm.RdBu_r.copy()
    cmap_rdbu.set_bad(color="lightgray", alpha=0.8)

    for ch_idx in range(3):
        orig = orig_show[ch_idx]
        pre = pre_show[ch_idx]
        fine = fine_show[ch_idx]
        diff_pre = np.abs(original_3ch[ch_idx] - recon_pretrained[ch_idx])
        diff_fine = np.abs(original_3ch[ch_idx] - recon_finetuned_3ch[ch_idx])
        if mask is not None:
            diff_pre = np.where(valid, diff_pre, np.nan)
            diff_fine = np.where(valid, diff_fine, np.nan)
        vmin = np.nanmin([orig, pre, fine])
        vmax = np.nanmax([orig, pre, fine])
        diff_max = max(np.nanmax(np.abs(diff_pre)), np.nanmax(np.abs(diff_fine)), 1e-6)
        im0 = axes[ch_idx, 0].imshow(orig, cmap=cmap_jet, vmin=vmin, vmax=vmax)
        axes[ch_idx, 0].set_title(f"{channel_names[ch_idx]} - Original", fontweight="bold")
        axes[ch_idx, 0].axis("off")
        plt.colorbar(im0, ax=axes[ch_idx, 0], fraction=0.046, pad=0.04)
        im1 = axes[ch_idx, 1].imshow(pre, cmap=cmap_jet, vmin=vmin, vmax=vmax)
        axes[ch_idx, 1].set_title(f"{channel_names[ch_idx]} - Pretrained", fontweight="bold")
        axes[ch_idx, 1].axis("off")
        plt.colorbar(im1, ax=axes[ch_idx, 1], fraction=0.046, pad=0.04)
        im2 = axes[ch_idx, 2].imshow(fine, cmap=cmap_jet, vmin=vmin, vmax=vmax)
        axes[ch_idx, 2].set_title(f"{channel_names[ch_idx]} - Finetuned", fontweight="bold")
        axes[ch_idx, 2].axis("off")
        plt.colorbar(im2, ax=axes[ch_idx, 2], fraction=0.046, pad=0.04)
        # diff_plot = np.where((diff_fine - diff_pre)<0, np.nan, diff_fine - diff_pre)
        diff_plot = diff_fine - diff_pre
        im3 = axes[ch_idx, 3].imshow(diff_plot, cmap=cmap_rdbu, vmin=-20, vmax=20)
        axes[ch_idx, 3].set_title(f"{channel_names[ch_idx]} - Error Diff", fontweight="bold")
        axes[ch_idx, 3].axis("off")
        plt.colorbar(im3, ax=axes[ch_idx, 3], fraction=0.046, pad=0.04)
        if mask is not None:
            mse_pre = np.nanmean((np.where(valid, original_3ch[ch_idx] - recon_pretrained[ch_idx], np.nan)) ** 2)
            mse_fine = np.nanmean((np.where(valid, original_3ch[ch_idx] - recon_finetuned_3ch[ch_idx], np.nan)) ** 2)
        else:
            mse_pre = np.mean((original_3ch[ch_idx] - recon_pretrained[ch_idx]) ** 2)
            mse_fine = np.mean((original_3ch[ch_idx] - recon_finetuned_3ch[ch_idx]) ** 2)
        rmse_pre = float(np.sqrt(mse_pre)) if np.isfinite(mse_pre) else np.nan
        rmse_fine = float(np.sqrt(mse_fine)) if np.isfinite(mse_fine) else np.nan
        axes[ch_idx, 1].text(0.02, 0.98, f"RMSE: {rmse_pre:.4f}", transform=axes[ch_idx, 1].transAxes,
                             fontsize=10, verticalalignment="top",
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        axes[ch_idx, 2].text(0.02, 0.98, f"RMSE: {rmse_fine:.4f}", transform=axes[ch_idx, 2].transAxes,
                             fontsize=10, verticalalignment="top",
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.suptitle(f"{title_prefix} - Time {time_idx + 1}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_3ch_comparison_paper(original_3ch, recon_pretrained, recon_finetuned_3ch,
                               time_idx, output_path, title_prefix="",
                               observation_mask=None):
    """
    论文用对比图：只绘制有效观测点（observation_mask），无指标标注，无 Error Diff 列。

    Args:
        original_3ch: 原始数据 (3, H, W)
        recon_pretrained: 预训练模型重建 (3, H, W)
        recon_finetuned_3ch: 微调模型重建 (3, H, W)
        time_idx: 时间索引
        output_path: 输出路径
        title_prefix: 标题前缀
        observation_mask: 有效点掩码 (H, W)，1=有效，0=无效；为 None 时显示全部像素
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from skimage.metrics import structural_similarity as ssim_func

    channel_names = ['6.25μm', '7.42μm', '10.80μm']
    # 每行3列的子标题: (序号) 标题
    subplot_titles = [
        [f'(a) AGRI {channel_names[0]}', f'(b) Decoded {channel_names[0]} (Pretrain)', f'(c) Decoded {channel_names[0]} (Finetune)'],
        [f'(d) AGRI {channel_names[1]}', f'(e) Decoded {channel_names[1]} (Pretrain)', f'(f) Decoded {channel_names[1]} (Finetune)'],
        [f'(g) AGRI {channel_names[2]}', f'(h) Decoded {channel_names[2]} (Pretrain)', f'(i) Decoded {channel_names[2]} (Finetune)'],
    ]

    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Serif'

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.rcParams.update({'font.size': 14, 'font.weight': 'normal'})

    alpha = observation_mask.astype(np.float32) if observation_mask is not None else None
    vmin_list = [200, 200, 200]
    vmax_list = [270, 290, 300]

    def compute_metrics(ref, pred, mask):
        """在有效点上计算 MAE, SSIM, CC"""
        if mask is not None:
            valid = mask > 0
            r, p = ref[valid], pred[valid]
        else:
            valid = None
            r, p = ref.flatten(), pred.flatten()
        mae = np.mean(np.abs(r - p))
        cc = np.corrcoef(r, p)[0, 1] if np.std(r) > 1e-8 and np.std(p) > 1e-8 else 0.0
        data_range = float(r.max() - r.min())
        if data_range > 1e-8:
            # 用有效像素均值填充无效区域
            if valid is not None:
                ref_filled = np.where(valid, ref, np.mean(r))
                pred_filled = np.where(valid, pred, np.mean(p))
            else:
                ref_filled, pred_filled = ref, pred
            sv = ssim_func(ref_filled, pred_filled, data_range=data_range)
        else:
            sv = 1.0
        return mae, sv, cc

    for ch_idx in range(3):
        orig = original_3ch[ch_idx]
        pre  = recon_pretrained[ch_idx]
        fine = recon_finetuned_3ch[ch_idx]

        norm = Normalize(vmin=vmin_list[ch_idx], vmax=vmax_list[ch_idx])
        cmap = plt.get_cmap('jet')

        for col_idx, data in enumerate([orig, pre, fine]):
            ax = axes[ch_idx, col_idx]
            rgba = cmap(norm(data))
            if alpha is not None:
                rgba[..., 3] = alpha
            ax.imshow(rgba)
            ax.set_title(subplot_titles[ch_idx][col_idx], fontweight='bold', loc='left', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor('black')

            # Pretrained / Finetuned 列标注指标
            if col_idx in (1, 2):
                mae, sv, cc = compute_metrics(orig, data, alpha)
                ax.text(0.03, 0.97,
                        f'MAE={mae:.3f}\nSSIM={sv:.3f}\nCC={cc:.3f}',
                        transform=ax.transAxes, fontsize=11, fontweight='normal',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

            divider = make_axes_locatable(ax)
            cax_dummy = divider.append_axes('right', size='5%', pad=0.15)
            cax_dummy.set_visible(col_idx == 2)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax_dummy, extend='both')
        cb.set_label('Tb (K)', fontsize=13, fontweight='bold')
        cb.ax.tick_params(labelsize=12)
        for label in cb.ax.get_yticklabels():
            label.set_fontweight('normal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_9ch_comparison(original_9ch, recon_finetuned_9ch, time_idx, output_path,
                        title_prefix="", observation_mask=None, metrics_9ch=None):
    """
    9 通道对比图：6 行 × 3 列。
    行 0: CH07/08/09 Original
    行 1: CH07/08/09 Finetuned
    行 2: CH10/11/12 Original
    行 3: CH10/11/12 Finetuned
    行 4: CH13/14/15 Original
    行 5: CH13/14/15 Finetuned
    每列右侧 colorbar，同一通道 vmin/vmax 统一。
    """
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Serif'

    CH_NAMES_SHORT = [
        "3.75μm", "3.75μm", "6.25μm",
        "6.95μm", "7.42μm", "8.55μm",
        "10.80μm", "12.00μm", "13.3μm",
    ]
    # 子图序号 (a)-(r), 6行×3列=18个
    labels = [chr(ord('a') + i) for i in range(18)]

    cmap = plt.cm.jet.copy()
    cmap.set_bad(color="lightgray", alpha=0.8)

    fig, axes = plt.subplots(6, 3, figsize=(10.5, 18),
                             gridspec_kw={"wspace": 0.02, "hspace": 0.25})
    plt.rcParams.update({"font.size": 11, "font.weight": "normal"})

    alpha = observation_mask.astype(np.float32) if observation_mask is not None else None

    # 预计算每个通道的 vmin/vmax（仅用有效观测）
    norms = []
    for ci in range(9):
        orig, fine = original_9ch[ci], recon_finetuned_9ch[ci]
        if alpha is not None:
            valid = alpha > 0
            all_vals = np.concatenate([orig[valid], fine[valid]])
        else:
            all_vals = np.concatenate([orig.flatten(), fine.flatten()])
        all_vals = all_vals[~np.isnan(all_vals)]
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        if ci >= 2:  # CH09-CH15: 固定最小值150
            vmin = 170
        norms.append(plt.Normalize(vmin=vmin, vmax=vmax))

    ims = {}

    for row in range(6):
        group = row // 2
        is_finetuned = row % 2
        row_label = "Decoded" if is_finetuned else "AGRI"

        for col in range(3):
            ci = group * 3 + col
            data = recon_finetuned_9ch[ci] if is_finetuned else original_9ch[ci]
            norm = norms[ci]
            idx = row * 3 + col

            ax = axes[row, col]
            rgba = cmap(norm(data))
            if alpha is not None:
                rgba[..., 3] = alpha
            ax.imshow(rgba)
            ax.set_title(f"({labels[idx]}) {row_label} {CH_NAMES_SHORT[ci]}",
                         fontsize=11, fontweight="bold", loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_edgecolor("black")

            if is_finetuned:
                ims[ci] = (ax, norm)
                if metrics_9ch and AGRI_CHANNEL_NAMES[ci] in metrics_9ch:
                    m = metrics_9ch[AGRI_CHANNEL_NAMES[ci]]
                    if m:
                        txt = f"MAE={m['mae']:.3f}\nSSIM={m['ssim']:.3f}\nCC={m['correlation']:.3f}"
                        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=9,
                                fontweight="normal", ha="left", va="top",
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75))

    # Finetuned 行加 colorbar
    for ci, (ax, norm) in ims.items():
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, extend="both")
        cb.set_label("Tb (K)", fontsize=10, fontweight="bold")
        cb.ax.tick_params(labelsize=9)
        for label in cb.ax.get_yticklabels():
            label.set_fontweight('normal')

    # Original 行加不可见占位 colorbar 对齐宽度
    for row in range(0, 6, 2):
        for col in range(3):
            ci = (row // 2) * 3 + col
            norm = norms[ci]
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=axes[row, col], fraction=0.046, pad=0.02)
            cb.ax.set_visible(False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_time_range(start_str: str, end_str: str) -> Tuple[dt.datetime, dt.datetime]:
    def parse_any(s: str) -> dt.datetime:
        for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y%m%d%H%M%S", "%Y%m%d"]:
            try:
                return dt.datetime.strptime(s.strip(), fmt)
            except ValueError:
                continue
        raise ValueError(f"无法解析时间: {s}")
    start_dt = parse_any(start_str)
    end_dt = parse_any(end_str)
    if end_dt <= start_dt:
        raise ValueError("结束时间必须晚于开始时间")
    return start_dt, end_dt


def _build_eval_tag(finetuned_iter: int, tokenizer_label: str = "") -> str:
    prefix = f"{tokenizer_label}_" if tokenizer_label else ""
    return f"{prefix}iter{finetuned_iter}_h{H0}-{H1}_w{W0}-{W1}"


def _bucket_to_stats(bucket):
    stats = {}
    for k, vals in bucket.items():
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            stats[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
    return stats


def _write_stats_block(f, title: str, stats, metric_order):
    f.write(f"\n{title}\n")
    if not stats:
        f.write("  (无有效数据)\n")
        return
    for name in metric_order:
        if name in stats:
            v = stats[name]
            f.write(f"  {name.upper():12s}: {v['mean']:.6f} ± {v['std']:.6f}\n")


def _collect_metric_buckets(results: List[dict], report_both_metrics: bool):
    metric_keys = ["mse", "mae", "rmse", "bias", "correlation", "psnr", "ssim"]
    buckets = {
        "pretrained_3ch": {k: [] for k in metric_keys},
        "finetuned_3ch": {k: [] for k in metric_keys},
        "finetuned_all_ch": {k: [] for k in metric_keys},
        "pretrained_3ch_all": {k: [] for k in metric_keys},
        "finetuned_3ch_all": {k: [] for k in metric_keys},
        "finetuned_all_ch_all": {k: [] for k in metric_keys},
    }
    key_map = {
        "pretrained": "pretrained_3ch",
        "finetuned_3ch": "finetuned_3ch",
        "finetuned": "finetuned_all_ch",
        "pretrained_all": "pretrained_3ch_all",
        "finetuned_3ch_all": "finetuned_3ch_all",
        "finetuned_all": "finetuned_all_ch_all",
    }
    for result in results:
        for result_key, bucket_key in key_map.items():
            if result_key.endswith("_all") and not report_both_metrics:
                continue
            metric = result.get(result_key)
            if not metric:
                continue
            for name in metric_keys:
                if name in metric and metric[name] is not None:
                    buckets[bucket_key][name].append(metric[name])
    summary = {
        "valid_only": {
            "pretrained_3ch": _bucket_to_stats(buckets["pretrained_3ch"]),
            "finetuned_3ch": _bucket_to_stats(buckets["finetuned_3ch"]),
            "finetuned_all_ch": _bucket_to_stats(buckets["finetuned_all_ch"]),
        }
    }
    if report_both_metrics:
        summary["all_pixels"] = {
            "pretrained_3ch": _bucket_to_stats(buckets["pretrained_3ch_all"]),
            "finetuned_3ch": _bucket_to_stats(buckets["finetuned_3ch_all"]),
            "finetuned_all_ch": _bucket_to_stats(buckets["finetuned_all_ch_all"]),
        }
    return summary


def _metric_mean(summary: dict, section: str, model_key: str, metric: str) -> float:
    return summary.get(section, {}).get(model_key, {}).get(metric, {}).get("mean", float("nan"))


def run_compare_agri_latent16_latent32(args) -> None:
    if args.output_dir == "outputs/evaluation_results_agri_region":
        args.output_dir = AGRI_LATENT_COMPARE_OUTPUT_DIR
    os.makedirs(args.output_dir, exist_ok=True)

    min_vals, max_vals = None, None
    if args.stats_path and os.path.exists(args.stats_path):
        min_vals, max_vals = load_channel_stats(args.stats_path)
        print(f"已加载统计量: {args.stats_path}")
    elif args.stats_path:
        print(f"警告: 统计量文件不存在: {args.stats_path}")

    tokenizer_specs = [
        {
            "label": "agri_latent16",
            "dir": args.latent16_dir,
            "iter": args.latent16_iter,
        },
        {
            "label": "agri_latent32",
            "dir": args.latent32_dir,
            "iter": args.latent32_iter,
        },
    ]

    print("=" * 60)
    print("AGRI Tokenizer 对比评估（latent16 vs latent32）")
    print("=" * 60)
    print(f"裁剪区域: h=[{H0}:{H1}], w=[{W0}:{W1}] ({REGION_H}x{REGION_W})")
    print(f"数据目录: {args.input_dir}")
    print(f"预训练模型: {args.pretrained_dir}")
    for spec in tokenizer_specs:
        print(f"{spec['label']}: {spec['dir']}  iter={spec['iter']}")
    print(f"输出目录: {args.output_dir}")
    print(f"时间范围: {args.start} ~ {args.end}  步长: {args.step_minutes} 分钟")
    if args.save_plots:
        print(f"保存对比图: 是（每个 tokenizer 最多 {args.plot_samples} 个样本）")
    print("=" * 60)

    print("\n加载预训练模型...")
    pretrained_model = VideoTokenizerWrapper(
        args.pretrained_dir, iter_num=None, device=args.device, dtype=args.dtype
    )
    finetuned_models = {}
    for spec in tokenizer_specs:
        print(f"加载 {spec['label']}...")
        finetuned_models[spec["label"]] = VideoTokenizerWrapper(
            spec["dir"], iter_num=spec["iter"], device=args.device, dtype=args.dtype
        )

    start_dt, end_dt = parse_time_range(args.start, args.end)
    ref_times = list(iter_reference_times(start_dt, end_dt, args.step_minutes))
    if args.max_samples is not None:
        ref_times = ref_times[: args.max_samples]
    print(f"\n待评估参考时间数: {len(ref_times)}")

    results_by_label = {spec["label"]: [] for spec in tokenizer_specs}
    skipped_by_label = {spec["label"]: 0 for spec in tokenizer_specs}
    plot_count_by_label = {spec["label"]: 0 for spec in tokenizer_specs}

    for ref_dt in tqdm(ref_times, desc="对比评估中"):
        for spec in tokenizer_specs:
            label = spec["label"]
            need_plot = args.save_plots and plot_count_by_label[label] < args.plot_samples
            result = process_one_reference_time(
                ref_dt,
                args.input_dir,
                pretrained_model,
                finetuned_models[label],
                min_vals,
                max_vals,
                use_mask_channel=args.use_mask_channel,
                return_arrays_for_plot=need_plot,
                report_both_metrics=args.report_both_metrics,
                min_valid_frames=args.min_valid_frames,
                min_valid_ratio_per_frame=args.min_valid_ratio,
            )
            if result is None:
                skipped_by_label[label] += 1
                continue

            result["tokenizer_label"] = label
            result["finetuned_dir"] = spec["dir"]
            result["finetuned_iter"] = spec["iter"]

            if "_plot_arrays" in result:
                orig_plot, pre_plot, fine_plot, mask_plot = result.pop("_plot_arrays")
                plot_dir = os.path.join(args.output_dir, "plots", label, result["ref_time"])
                os.makedirs(plot_dir, exist_ok=True)
                for t in range(orig_plot.shape[1]):
                    plot_path = os.path.join(plot_dir, f"comparison_3ch_{label}_iter{spec['iter']}_t{t + 1}.png")
                    plot_3ch_comparison_paper(
                        orig_plot[:, t, :, :],
                        pre_plot[:, t, :, :],
                        fine_plot[:, t, :, :],
                        time_idx=t,
                        output_path=plot_path,
                        title_prefix=f"{result['ref_time']} {label}",
                        observation_mask=mask_plot[t] if mask_plot is not None else None,
                    )
                plot_count_by_label[label] += 1

            if "_plot_9ch_arrays" in result:
                orig_9ch, fine_9ch, mask_9ch = result.pop("_plot_9ch_arrays")
                plot_dir = os.path.join(args.output_dir, "plots", label, result["ref_time"])
                os.makedirs(plot_dir, exist_ok=True)
                for t in range(orig_9ch.shape[1]):
                    plot_path = os.path.join(plot_dir, f"comparison_9ch_{label}_iter{spec['iter']}_t{t + 1}.png")
                    plot_9ch_comparison(
                        orig_9ch[:, t, :, :],
                        fine_9ch[:, t, :, :],
                        time_idx=t,
                        output_path=plot_path,
                        title_prefix=f"{result['ref_time']} {label}",
                        observation_mask=mask_9ch[t] if mask_9ch is not None else None,
                        metrics_9ch=result.get("finetuned_per_channel"),
                    )

            results_by_label[label].append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries = {
        label: _collect_metric_buckets(results, args.report_both_metrics)
        for label, results in results_by_label.items()
    }

    print("\n" + "=" * 60)
    print("latent16 vs latent32 对比（仅有效观测区）")
    print("=" * 60)
    for label in results_by_label:
        n = len(results_by_label[label])
        rmse_3ch = _metric_mean(summaries[label], "valid_only", "finetuned_3ch", "rmse")
        rmse_9ch = _metric_mean(summaries[label], "valid_only", "finetuned_all_ch", "rmse")
        mae_9ch = _metric_mean(summaries[label], "valid_only", "finetuned_all_ch", "mae")
        ssim_9ch = _metric_mean(summaries[label], "valid_only", "finetuned_all_ch", "ssim")
        print(
            f"{label}: samples={n}, 3ch RMSE={rmse_3ch:.6f}, "
            f"9ch RMSE={rmse_9ch:.6f}, 9ch MAE={mae_9ch:.6f}, 9ch SSIM={ssim_9ch:.6f}"
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        args.output_dir,
        f"evaluation_region_agri_latent16_vs_latent32_iter{args.latent16_iter}_{timestamp}.json",
    )
    report_file = os.path.join(
        args.output_dir,
        f"evaluation_report_agri_latent16_vs_latent32_iter{args.latent16_iter}_{timestamp}.txt",
    )

    summary = {
        "config": {
            "crop_region": {"H0": H0, "H1": H1, "W0": W0, "W1": W1, "REGION_H": REGION_H, "REGION_W": REGION_W},
            "offsets_minutes": AGRI_VIDEO_OFFSETS_MINUTES,
            "agri_channels": AGRI_CHANNEL_NAMES,
            "pretrained_compare_channels": {
                "indices": PRETRAINED_CHANNELS,
                "names": [AGRI_CHANNEL_NAMES[i] for i in PRETRAINED_CHANNELS],
            },
            "input_dir": args.input_dir,
            "pretrained_dir": args.pretrained_dir,
            "tokenizers": tokenizer_specs,
            "device": args.device,
            "dtype": args.dtype,
            "max_samples": args.max_samples,
            "stats_path": args.stats_path,
            "stats_loaded": bool(min_vals is not None and max_vals is not None),
            "use_mask_channel": args.use_mask_channel,
            "save_plots": args.save_plots,
            "plot_samples": args.plot_samples,
            "report_both_metrics": args.report_both_metrics,
            "min_valid_frames": args.min_valid_frames,
            "min_valid_ratio": args.min_valid_ratio,
            "skipped_by_label": skipped_by_label,
            "start": args.start,
            "end": args.end,
            "step_minutes": args.step_minutes,
            "num_reference_times": len(ref_times),
            "num_samples_by_label": {label: len(results) for label, results in results_by_label.items()},
        },
        "summary_by_label": summaries,
        "details_by_label": results_by_label,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    metric_order = ["mse", "mae", "rmse", "bias", "correlation", "psnr", "ssim"]
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("AGRI Tokenizer latent16 vs latent32 区域评估报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"报告时间: {timestamp}\n")
        f.write(f"裁剪区域: h=[{H0}:{H1}], w=[{W0}:{W1}] ({REGION_H}x{REGION_W})\n")
        f.write(f"时间范围: {args.start} ~ {args.end}, step={args.step_minutes} min\n")
        f.write(f"预训练模型: {args.pretrained_dir}\n")
        for spec in tokenizer_specs:
            f.write(f"{spec['label']}: {spec['dir']} iter={spec['iter']}\n")
        f.write(f"成功样本数: { {label: len(results) for label, results in results_by_label.items()} }\n")
        f.write(f"跳过样本数: {skipped_by_label}\n")
        for label, summary_one in summaries.items():
            f.write("\n" + "-" * 60 + "\n")
            f.write(f"{label}\n")
            _write_stats_block(
                f,
                "[仅有效观测区] 微调 3 通道（CH09/CH11/CH13）",
                summary_one["valid_only"]["finetuned_3ch"],
                metric_order,
            )
            _write_stats_block(
                f,
                "[仅有效观测区] 微调 9 通道（CH07-CH15）",
                summary_one["valid_only"]["finetuned_all_ch"],
                metric_order,
            )
            if args.report_both_metrics and "all_pixels" in summary_one:
                _write_stats_block(
                    f,
                    "[全图（含无效区）] 微调 9 通道（CH07-CH15）",
                    summary_one["all_pixels"]["finetuned_all_ch"],
                    metric_order,
                )

    print(f"\n详细结果已保存: {results_file}")
    print(f"评估报告已保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="在新裁剪区域上批量评估 AGRI 视频 Tokenizer（与 prepare_agri_video_tokens 区域一致）"
    )
    parser.add_argument("--start", required=True, help="开始时间，如 2024-07-01 或 2024-07-01 00:00")
    parser.add_argument("--end", required=True, help="结束时间，如 2024-07-02")
    parser.add_argument("--input_dir", default="/public/share/users/sunhaofei/yyf_data/AGRI_processed",
                        help="AGRI_processed 根目录，下为 YYYYMMDDHHMM_agri.npz")
    parser.add_argument("--pretrained_dir", default="/public/home/sunhaofei/cosmos-predict1/checkpoints/Cosmos-Tokenize1-CV4x8x8-360p",
                        help="预训练模型目录")
    parser.add_argument("--finetuned_dir", default=None, help="微调模型 checkpoint 目录")
    parser.add_argument("--finetuned_iter", type=int, default=210000, help="微调模型迭代次数")
    parser.add_argument("--output_dir", default="outputs/evaluation_results_agri_region", help="输出目录")
    parser.add_argument("--eval_agri_latent32", action="store_true",
                        help="评估 2026-05-13 训练的 AGRI tokenizer（latent 通道 32，默认 iter=210000）")
    parser.add_argument("--compare_agri_latent16_latent32", action="store_true",
                        help="同时评估并比较 latent16 旧版和 latent32 新版 AGRI tokenizer")
    parser.add_argument("--latent16_dir", default=AGRI_LATENT16_FINETUNED_DIR,
                        help="latent16 AGRI tokenizer checkpoint 目录")
    parser.add_argument("--latent16_iter", type=int, default=AGRI_LATENT16_ITER,
                        help="latent16 AGRI tokenizer 迭代次数")
    parser.add_argument("--latent32_dir", default=AGRI_LATENT32_FINETUNED_DIR,
                        help="latent32 AGRI tokenizer checkpoint 目录")
    parser.add_argument("--latent32_iter", type=int, default=AGRI_LATENT32_ITER,
                        help="latent32 AGRI tokenizer 迭代次数")
    parser.add_argument("--tokenizer_label", default="",
                        help="写入结果文件名和报告 config 的 tokenizer 标签")
    parser.add_argument("--device", default="cuda", help="计算设备")
    parser.add_argument("--dtype", default="bfloat16", help="数据类型")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数（用于测试）")
    parser.add_argument("--step_minutes", type=int, default=15, help="参考时间步长（分钟）")
    parser.add_argument("--use_mask_channel", action="store_true", default=True, help="微调模型使用 mask 通道")
    parser.add_argument("--no_mask_channel", action="store_true", help="微调模型不使用 mask 通道")
    parser.add_argument("--stats_path", default=STATS_PATH, help="通道 min/max 统计量路径")
    parser.add_argument("--save_plots", action="store_true", help="对部分样本保存 3 通道对比图（Original / Pretrained / Finetuned）")
    parser.add_argument("--plot_samples", type=int, default=5, help="绘制对比图的样本数量（默认 5）")
    parser.add_argument("--report_both_metrics", action="store_true",
                        help="同时输出「仅有效区」与「全图（含无效区）」指标，便于分析无效区对 MSE 的影响")
    parser.add_argument("--min_valid_frames", type=int, default=5,
                        help="5 帧中至少有多少帧的有效观测比例达到阈值才纳入评估，否则视为异常样本跳过（默认 2）")
    parser.add_argument("--min_valid_ratio", type=float, default=0.5,
                        help="单帧有效观测比例阈值，与 --min_valid_frames 配合（默认 0.5）")
    args = parser.parse_args()

    if args.no_mask_channel:
        args.use_mask_channel = False

    if args.compare_agri_latent16_latent32:
        run_compare_agri_latent16_latent32(args)
        return

    if args.eval_agri_latent32:
        if args.finetuned_dir is None:
            args.finetuned_dir = AGRI_LATENT32_FINETUNED_DIR
        if args.finetuned_iter == parser.get_default("finetuned_iter"):
            args.finetuned_iter = AGRI_LATENT32_ITER
        if args.output_dir == parser.get_default("output_dir"):
            args.output_dir = AGRI_LATENT32_OUTPUT_DIR
        if not args.tokenizer_label:
            args.tokenizer_label = "agri_latent32"

    if args.finetuned_dir is None:
        parser.error("--finetuned_dir is required unless --eval_agri_latent32 is used")

    min_vals, max_vals = None, None
    if args.stats_path and os.path.exists(args.stats_path):
        try:
            min_vals, max_vals = load_channel_stats(args.stats_path)
            print(f"已加载统计量: {args.stats_path}")
        except Exception as e:
            print(f"警告: 加载统计量失败: {e}")
    else:
        if args.stats_path:
            print(f"警告: 统计量文件不存在: {args.stats_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("AGRI Tokenizer 批量评估（新裁剪区域）")
    print("=" * 60)
    print(f"裁剪区域: h=[{H0}:{H1}], w=[{W0}:{W1}] ({REGION_H}x{REGION_W})")
    print(f"5 个时刻偏移（分钟）: {AGRI_VIDEO_OFFSETS_MINUTES}")
    print(f"数据目录: {args.input_dir}")
    print(f"预训练模型: {args.pretrained_dir}")
    print(f"微调模型: {args.finetuned_dir}  iter={args.finetuned_iter}")
    if args.tokenizer_label:
        print(f"Tokenizer 标签: {args.tokenizer_label}")
    print(f"输出目录: {args.output_dir}")
    print(f"时间范围: {args.start} ~ {args.end}  步长: {args.step_minutes} 分钟")
    if args.save_plots:
        print(f"保存对比图: 是（最多 {args.plot_samples} 个样本，每样本 5 帧）")
    if args.report_both_metrics:
        print("指标: 同时输出「仅有效区」与「全图」")
    print(f"异常样本过滤: 5 帧中至少 {args.min_valid_frames} 帧 valid_ratio >= {args.min_valid_ratio} 才纳入评估")
    print("=" * 60)

    print("\n加载预训练模型...")
    pretrained_model = VideoTokenizerWrapper(
        args.pretrained_dir, iter_num=None, device=args.device, dtype=args.dtype
    )
    print("加载微调模型...")
    finetuned_model = VideoTokenizerWrapper(
        args.finetuned_dir, iter_num=args.finetuned_iter, device=args.device, dtype=args.dtype
    )

    start_dt, end_dt = parse_time_range(args.start, args.end)
    ref_times = list(iter_reference_times(start_dt, end_dt, args.step_minutes))
    if args.max_samples is not None:
        ref_times = ref_times[: args.max_samples]
    print(f"\n待评估参考时间数: {len(ref_times)}")

    results = []
    plot_count = 0
    skipped_abnormal = 0
    for ref_dt in tqdm(ref_times, desc="评估中"):
        need_plot = args.save_plots and plot_count < args.plot_samples
        r = process_one_reference_time(
            ref_dt,
            args.input_dir,
            pretrained_model,
            finetuned_model,
            min_vals,
            max_vals,
            use_mask_channel=args.use_mask_channel,
            return_arrays_for_plot=need_plot,
            report_both_metrics=args.report_both_metrics,
            min_valid_frames=args.min_valid_frames,
            min_valid_ratio_per_frame=args.min_valid_ratio,
        )
        if r is None:
            skipped_abnormal += 1
            continue
        if "_plot_arrays" in r:
            orig_plot, pre_plot, fine_plot, mask_plot = r.pop("_plot_arrays")
            T_plot = orig_plot.shape[1]
            plot_dir = os.path.join(args.output_dir, "plots", r["ref_time"])
            os.makedirs(plot_dir, exist_ok=True)
            for t in range(T_plot):
                plot_path = os.path.join(plot_dir, f"comparison_3ch_iter{args.finetuned_iter}_t{t + 1}.png")
                plot_3ch_comparison_paper(
                    orig_plot[:, t, :, :],
                    pre_plot[:, t, :, :],
                    fine_plot[:, t, :, :],
                    time_idx=t,
                    output_path=plot_path,
                    title_prefix=r["ref_time"],
                    observation_mask=mask_plot[t] if mask_plot is not None else None,
                )
            plot_count += 1
        if "_plot_9ch_arrays" in r:
            orig_9ch, fine_9ch, mask_9ch = r.pop("_plot_9ch_arrays")
            T_plot = orig_9ch.shape[1]
            plot_dir = os.path.join(args.output_dir, "plots", r["ref_time"])
            os.makedirs(plot_dir, exist_ok=True)
            for t in range(T_plot):
                plot_path = os.path.join(plot_dir, f"comparison_9ch_iter{args.finetuned_iter}_t{t + 1}.png")
                plot_9ch_comparison(
                    orig_9ch[:, t, :, :],
                    fine_9ch[:, t, :, :],
                    time_idx=t,
                    output_path=plot_path,
                    title_prefix=r["ref_time"],
                    observation_mask=mask_9ch[t] if mask_9ch is not None else None,
                    metrics_9ch=r.get("finetuned_per_channel"),
                )
        results.append(r)
        if len(results) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if skipped_abnormal > 0:
        print(f"\n已跳过异常/缺失样本数: {skipped_abnormal}（缺 slot 或有效帧数不足）")
    if len(results) == 0:
        print("没有成功评估的样本")
        return

    # 汇总
    pretrained_metrics = {"mse": [], "mae": [], "rmse": [], "bias": [], "correlation": [], "psnr": [], "ssim": []}
    finetuned_metrics = {"mse": [], "mae": [], "rmse": [], "bias": [], "correlation": [], "psnr": [], "ssim": []}
    finetuned_3ch_metrics = {"mse": [], "mae": [], "rmse": [], "bias": [], "correlation": [], "psnr": [], "ssim": []}
    pretrained_all_metrics = {"mse": [], "mae": [], "rmse": [], "bias": [], "correlation": [], "psnr": [], "ssim": []}
    finetuned_all_metrics = {"mse": [], "mae": [], "rmse": [], "bias": [], "correlation": [], "psnr": [], "ssim": []}
    finetuned_3ch_all_metrics = {"mse": [], "mae": [], "rmse": [], "bias": [], "correlation": [], "psnr": [], "ssim": []}
    for r in results:
        if r.get("pretrained"):
            for k in pretrained_metrics:
                pretrained_metrics[k].append(r["pretrained"][k])
        if r.get("finetuned"):
            for k in finetuned_metrics:
                finetuned_metrics[k].append(r["finetuned"][k])
        if r.get("finetuned_3ch"):
            for k in finetuned_3ch_metrics:
                finetuned_3ch_metrics[k].append(r["finetuned_3ch"][k])
        if args.report_both_metrics:
            if r.get("pretrained_all"):
                for k in pretrained_all_metrics:
                    pretrained_all_metrics[k].append(r["pretrained_all"][k])
            if r.get("finetuned_all"):
                for k in finetuned_all_metrics:
                    finetuned_all_metrics[k].append(r["finetuned_all"][k])
            if r.get("finetuned_3ch_all"):
                for k in finetuned_3ch_all_metrics:
                    finetuned_3ch_all_metrics[k].append(r["finetuned_3ch_all"][k])

    print("\n" + "=" * 60)
    print("评估结果汇总（仅有效观测区）")
    print("=" * 60)
    print("\n预训练模型 (3 通道 CH09, CH11, CH13):")
    for name in ["mse", "mae", "rmse", "bias", "correlation", "psnr", "ssim"]:
        v = pretrained_metrics[name]
        if v:
            print(f"  {name.upper():12s}: {np.mean(v):.6f} ± {np.std(v):.6f}")
    print("\n微调模型 (同样 3 通道):")
    for name in ["mse", "mae", "rmse", "bias", "correlation", "psnr", "ssim"]:
        v = finetuned_3ch_metrics[name]
        if v:
            print(f"  {name.upper():12s}: {np.mean(v):.6f} ± {np.std(v):.6f}")
    print("\n微调模型 (全部 9 通道):")
    for name in ["mse", "mae", "rmse", "bias", "correlation", "psnr", "ssim"]:
        v = finetuned_metrics[name]
        if v:
            print(f"  {name.upper():12s}: {np.mean(v):.6f} ± {np.std(v):.6f}")

    print("\n改进对比 (3 通道, 仅有效区):")
    for name in ["mse", "mae", "rmse"]:
        pre = np.mean(pretrained_metrics[name]) if pretrained_metrics[name] else 0
        fine = np.mean(finetuned_3ch_metrics[name]) if finetuned_3ch_metrics[name] else 0
        if pre > 0:
            print(f"  {name.upper():12s}: {pre:.6f} -> {fine:.6f} (改进 {(pre - fine) / pre * 100:+.2f}%)")
    for name in ["correlation", "psnr"]:
        pre = np.mean(pretrained_metrics[name]) if pretrained_metrics[name] else 0
        fine = np.mean(finetuned_3ch_metrics[name]) if finetuned_3ch_metrics[name] else 0
        print(f"  {name.upper():12s}: {pre:.6f} -> {fine:.6f} (改进 {fine - pre:+.6f})")

    # 逐通道汇总
    def aggregate_per_key(results_list, result_key, sub_keys, metric_keys):
        """从 results 列表中聚合 result[result_key][sub_key][metric] 的均值±标准差"""
        buckets = {sk: {mk: [] for mk in metric_keys} for sk in sub_keys}
        for r in results_list:
            per = r.get(result_key, {}) or {}
            for sk in sub_keys:
                m = per.get(sk)
                if m:
                    for mk in metric_keys:
                        if mk in m and m[mk] is not None:
                            buckets[sk][mk].append(m[mk])
        return {sk: _bucket_to_stats(buckets[sk]) for sk in sub_keys}

    metric_order = ["mse", "mae", "rmse", "bias", "correlation", "psnr", "ssim"]
    frame_keys = [f"t{ti}_offset{AGRI_VIDEO_OFFSETS_MINUTES[ti]:+d}min" for ti in range(NUM_AGRI_VIDEO_FRAMES)]
    pretrained_ch_names = [AGRI_CHANNEL_NAMES[i] for i in PRETRAINED_CHANNELS]

    per_ch_pretrained = aggregate_per_key(results, "pretrained_per_channel", pretrained_ch_names, metric_order)
    per_ch_finetuned  = aggregate_per_key(results, "finetuned_per_channel",  AGRI_CHANNEL_NAMES,   metric_order)
    per_frame_pretrained = aggregate_per_key(results, "pretrained_per_frame", frame_keys, metric_order)
    per_frame_finetuned  = aggregate_per_key(results, "finetuned_per_frame",  frame_keys, metric_order)

    print("\n逐通道 RMSE（预训练 vs 微调）:")
    for ch in pretrained_ch_names:
        pre_rmse = per_ch_pretrained[ch].get("rmse", {}).get("mean", float("nan"))
        fine_rmse = per_ch_finetuned[ch].get("rmse", {}).get("mean", float("nan"))
        print(f"  {ch:6s}: pretrained={pre_rmse:.6f}  finetuned={fine_rmse:.6f}")
    for ch in [c for c in AGRI_CHANNEL_NAMES if c not in pretrained_ch_names]:
        fine_rmse = per_ch_finetuned[ch].get("rmse", {}).get("mean", float("nan"))
        print(f"  {ch:6s}: pretrained=N/A           finetuned={fine_rmse:.6f}")

    print("\n逐帧 RMSE（预训练 vs 微调，3ch 可比通道）:")
    for fk in frame_keys:
        pre_rmse = per_frame_pretrained[fk].get("rmse", {}).get("mean", float("nan"))
        fine_rmse = per_frame_finetuned[fk].get("rmse", {}).get("mean", float("nan"))
        print(f"  {fk}: pretrained={pre_rmse:.6f}  finetuned={fine_rmse:.6f}")
        print("\n" + "=" * 60)
        print("评估结果汇总（全图，含无效区）")
        print("=" * 60)
        print("\n预训练 (3ch) 全图:")
        for name in ["mse", "mae", "rmse", "correlation", "psnr"]:
            v = pretrained_all_metrics[name]
            if v:
                print(f"  {name.upper():12s}: {np.mean(v):.6f} ± {np.std(v):.6f}")
        print("\n微调 (3ch) 全图:")
        for name in ["mse", "mae", "rmse", "correlation", "psnr"]:
            v = finetuned_3ch_all_metrics[name]
            if v:
                print(f"  {name.upper():12s}: {np.mean(v):.6f} ± {np.std(v):.6f}")
        print("\n改进对比 (3 通道, 全图):")
        for name in ["mse", "mae", "rmse"]:
            pre = np.mean(pretrained_all_metrics[name]) if pretrained_all_metrics[name] else 0
            fine = np.mean(finetuned_3ch_all_metrics[name]) if finetuned_3ch_all_metrics[name] else 0
            if pre > 0:
                print(f"  {name.upper():12s}: {pre:.6f} -> {fine:.6f} (改进 {(pre - fine) / pre * 100:+.2f}%)")
        print("  (全图 MSE 常因无效区重建差异而变差，有效区指标更能反映观测重建质量)")

    metric_order = ["mse", "mae", "rmse", "bias", "correlation", "psnr", "ssim"]
    valid_pretrained_stats = _bucket_to_stats(pretrained_metrics)
    valid_finetuned_3ch_stats = _bucket_to_stats(finetuned_3ch_metrics)
    valid_finetuned_all_ch_stats = _bucket_to_stats(finetuned_metrics)
    all_pretrained_stats = _bucket_to_stats(pretrained_all_metrics) if args.report_both_metrics else {}
    all_finetuned_3ch_stats = _bucket_to_stats(finetuned_3ch_all_metrics) if args.report_both_metrics else {}
    all_finetuned_all_ch_stats = _bucket_to_stats(finetuned_all_metrics) if args.report_both_metrics else {}
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_tag = _build_eval_tag(args.finetuned_iter, args.tokenizer_label)
    results_file = os.path.join(args.output_dir, f"evaluation_region_{eval_tag}_{timestamp}.json")
    report_file = os.path.join(args.output_dir, f"evaluation_report_{eval_tag}_{timestamp}.txt")
    pretrained_channel_names = [AGRI_CHANNEL_NAMES[i] for i in PRETRAINED_CHANNELS]

    summary = {
        "config": {
            "crop_region": {"H0": H0, "H1": H1, "W0": W0, "W1": W1, "REGION_H": REGION_H, "REGION_W": REGION_W},
            "offsets_minutes": AGRI_VIDEO_OFFSETS_MINUTES,
            "agri_channels": AGRI_CHANNEL_NAMES,
            "pretrained_compare_channels": {
                "indices": PRETRAINED_CHANNELS,
                "names": pretrained_channel_names,
            },
            "input_dir": args.input_dir,
            "pretrained_dir": args.pretrained_dir,
            "finetuned_dir": args.finetuned_dir,
            "finetuned_iter": args.finetuned_iter,
            "tokenizer_label": args.tokenizer_label,
            "eval_agri_latent32": args.eval_agri_latent32,
            "device": args.device,
            "dtype": args.dtype,
            "max_samples": args.max_samples,
            "stats_path": args.stats_path,
            "stats_loaded": bool(min_vals is not None and max_vals is not None),
            "use_mask_channel": args.use_mask_channel,
            "save_plots": args.save_plots,
            "plot_samples": args.plot_samples,
            "report_both_metrics": args.report_both_metrics,
            "min_valid_frames": args.min_valid_frames,
            "min_valid_ratio": args.min_valid_ratio,
            "skipped_abnormal": skipped_abnormal,
            "start": args.start,
            "end": args.end,
            "step_minutes": args.step_minutes,
            "num_reference_times": len(ref_times),
            "num_samples": len(results),
        },
        "summary": {
            "valid_only": {
                "pretrained_3ch": valid_pretrained_stats,
                "finetuned_3ch": valid_finetuned_3ch_stats,
                "finetuned_all_ch": valid_finetuned_all_ch_stats,
            },
            "per_channel": {
                "pretrained": per_ch_pretrained,
                "finetuned": per_ch_finetuned,
            },
            "per_frame": {
                "pretrained": per_frame_pretrained,
                "finetuned": per_frame_finetuned,
            },
        },
        "details": results,
    }
    if args.report_both_metrics and all_pretrained_stats:
        summary["summary"]["all_pixels"] = {
            "pretrained_3ch": all_pretrained_stats,
            "finetuned_3ch": all_finetuned_3ch_stats,
            "finetuned_all_ch": all_finetuned_all_ch_stats,
        }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存: {results_file}")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("AGRI Tokenizer 区域评估报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"报告时间: {timestamp}\n")
        f.write(f"裁剪区域: h=[{H0}:{H1}], w=[{W0}:{W1}] ({REGION_H}x{REGION_W})\n")
        f.write(f"5 个时刻偏移（分钟）: {AGRI_VIDEO_OFFSETS_MINUTES}\n")
        f.write(f"时间范围: {args.start} ~ {args.end}\n")
        f.write(f"步长采样: {args.step_minutes} 分钟\n")
        f.write(f"输入目录: {args.input_dir}\n")
        f.write(f"预训练模型: {args.pretrained_dir}\n")
        f.write(f"微调模型: {args.finetuned_dir}\n")
        f.write(f"微调迭代: {args.finetuned_iter}\n")
        f.write(f"预训练对比通道: idx={PRETRAINED_CHANNELS}, names={pretrained_channel_names}\n")
        f.write(f"设备/精度: {args.device} / {args.dtype}\n")
        f.write(f"统计量文件: {args.stats_path}\n")
        f.write(f"输出目录: {args.output_dir}\n")
        f.write(f"mask 通道: {'是' if args.use_mask_channel else '否'}\n")
        f.write(f"保存对比图: {'是' if args.save_plots else '否'}")
        if args.save_plots:
            f.write(f"（最多 {args.plot_samples} 个样本）\n")
        else:
            f.write("\n")
        f.write(f"指标模式: {'仅有效区 + 全图' if args.report_both_metrics else '仅有效区'}\n")
        f.write(
            f"异常样本过滤: 5 帧中至少 {args.min_valid_frames} 帧 valid_ratio >= {args.min_valid_ratio}\n"
        )
        f.write(f"候选参考时间数: {len(ref_times)}\n")
        f.write(f"成功评估样本数: {len(results)}\n")
        f.write(f"跳过样本数: {skipped_abnormal}\n")

        _write_stats_block(f, "[仅有效观测区] 预训练 3 通道（CH09/CH11/CH13）", valid_pretrained_stats, metric_order)
        _write_stats_block(f, "[仅有效观测区] 微调 3 通道（CH09/CH11/CH13）", valid_finetuned_3ch_stats, metric_order)
        _write_stats_block(f, "[仅有效观测区] 微调 9 通道（CH07-CH15）", valid_finetuned_all_ch_stats, metric_order)

        f.write("\n改进对比 (3 通道, 仅有效区):\n")
        for name in ["mse", "mae", "rmse"]:
            if name in valid_pretrained_stats and name in valid_finetuned_3ch_stats:
                pre = valid_pretrained_stats[name]["mean"]
                fine = valid_finetuned_3ch_stats[name]["mean"]
                if pre > 0:
                    f.write(f"  {name.upper():12s}: {pre:.6f} -> {fine:.6f} (改进 {(pre - fine) / pre * 100:+.2f}%)\n")
        for name in ["correlation", "psnr"]:
            if name in valid_pretrained_stats and name in valid_finetuned_3ch_stats:
                pre = valid_pretrained_stats[name]["mean"]
                fine = valid_finetuned_3ch_stats[name]["mean"]
                f.write(f"  {name.upper():12s}: {pre:.6f} -> {fine:.6f} (改进 {fine - pre:+.6f})\n")

        if args.report_both_metrics and all_pretrained_stats:
            _write_stats_block(f, "[全图（含无效区）] 预训练 3 通道", all_pretrained_stats, metric_order)
            _write_stats_block(f, "[全图（含无效区）] 微调 3 通道", all_finetuned_3ch_stats, metric_order)
            _write_stats_block(f, "[全图（含无效区）] 微调 9 通道", all_finetuned_all_ch_stats, metric_order)
            f.write("\n改进对比 (3 通道, 全图):\n")
            for name in ["mse", "mae", "rmse"]:
                if name in all_pretrained_stats and name in all_finetuned_3ch_stats:
                    pre = all_pretrained_stats[name]["mean"]
                    fine = all_finetuned_3ch_stats[name]["mean"]
                    if pre > 0:
                        f.write(f"  {name.upper():12s}: {pre:.6f} -> {fine:.6f} (改进 {(pre - fine) / pre * 100:+.2f}%)\n")
            for name in ["correlation", "psnr"]:
                if name in all_pretrained_stats and name in all_finetuned_3ch_stats:
                    pre = all_pretrained_stats[name]["mean"]
                    fine = all_finetuned_3ch_stats[name]["mean"]
                    f.write(f"  {name.upper():12s}: {pre:.6f} -> {fine:.6f} (改进 {fine - pre:+.6f})\n")

    print(f"评估报告已保存: {report_file}")


if __name__ == "__main__":
    main()
