import glob
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Tuple



def prepare_cref_dataset_auto(
        input_dir: str,
        output_dir: str,
        lat_range: Tuple[float, float] = (22.0, 28.4),
        lon_range: Tuple[float, float] = (110.0, 120.24)
) -> None:
    """
    自动生成时间窗口并处理 CREF 数据

    Args:
        input_dir: 输入 .nc 文件路径
        output_dir: 输出目录路径
        lat_range: 纬度范围 (min, max)
        lon_range: 经度范围 (min, max)
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据集并筛选空间范围
    # ds = xr.open_mfdataset(f"{input_dir}/*.nc", parallel=True)
    ds = xr.open_mfdataset(f"{input_dir}/*.nc", parallel=False)  # 关闭并行读取， 牺牲速度换稳定性
    print(ds)
    cref_da = ds['CREF'].sel(lat=slice(*lat_range), lon=slice(*lon_range))
    # cref_da = ds['CREF']
    print(f"原始数据形状: {cref_da.shape}")
    # # 最大最小值
    # print(f"最大值: {cref_da.max().values}, 最小值: {cref_da.min().values}, 平均值: {cref_da.mean().values}")
    cref_da = cref_da.fillna(0)

    # 验证时间维度
    times = cref_da.time.values
    if len(times) < 33:
        raise ValueError(f"数据时间步不足，需要至少33帧，当前仅{len(times)}帧")

    # 转换为 pandas 时间索引
    time_index = pd.DatetimeIndex(times)

    # 计算每小时包含的帧数
    delta_minutes = 6
    frames_per_hour = int(60 / delta_minutes)
    # print(f"时间间隔: {delta_minutes:.1f} 分钟/帧 | 每小时帧数: {frames_per_hour}")

    # 生成时间窗口（每小时滑动一次）
    window_size = 33
    time_windows = []
    for start_idx in range(0, len(time_index) - window_size + 1, frames_per_hour):
        window_times = time_index[start_idx: start_idx + window_size]
        # 验证窗口时间连续性
        time_diff = (window_times[-1] - window_times[0]).total_seconds() / 60
        expected_diff = delta_minutes * (window_size - 1)
        if not np.isclose(time_diff, expected_diff, atol=1):
            continue  # 跳过不连续窗口
        time_windows.append((
            window_times[0].isoformat(),
            window_times[-1].isoformat()
        ))

    print(f"生成 {len(time_windows)} 个有效时间窗口")

    # 处理每个时间窗口
    for i, (start_time, end_time) in enumerate(time_windows):
        try:
            # 选择时间切片
            ds_slice = cref_da.sel(time=slice(start_time, end_time))
            print(f"start_time is: {start_time}")

            # 数据预处理
            input_tensor = process_cref_data(ds_slice.values)
            # print(input_tensor.shape)
            # print(input_tensor.max())
            # print(input_tensor.min())
            # print(input_tensor.mean())
            # check 0-value ratio
            zero_ratio = np.mean(input_tensor == 0)
            print("in this sample, zero_ratio=", zero_ratio) 

            if zero_ratio > 0.65:
               continue 
            
            # 生成文件名
            start_str = pd.to_datetime(start_time).strftime("%Y%m%dT%H%M")
            output_path = Path(output_dir) / f"cref_{start_str}"

            # 标准化
            normalized_data = 2 * (input_tensor / 70) - 1
            np.savez_compressed(output_path, normalized_data)
            print(f"保存窗口 {i + 1}/{len(time_windows)}: {output_path.name}")

        except Exception as e:
            print(f"处理窗口 {start_time}-{end_time} 失败: {str(e)}")


def process_cref_data(raw_data: np.ndarray):
    """数据预处理流水线"""
    # 数据验证
    if raw_data.ndim != 3 or raw_data.shape != (33, 640, 1024):
        raise ValueError(f"输入数据形状异常: {raw_data.shape}，应为 (33, 640, 1024)")

    # 处理流程
    data = raw_data.astype(np.float32)
    data = np.clip(data, 0, 70)
    data_3c = np.repeat(data[:, np.newaxis, :, :], 3, axis=1)  # [T, C, H, W]
    data_3c = data_3c.transpose(1, 0, 2, 3)  # [C, T, H, W]
    return data_3c

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./datasets/CREF_Input/202306/')
    parser.add_argument('--output_dir', type=str, default="./auto_processed")
    args = parser.parse_args()
    
    prepare_cref_dataset_auto(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lat_range=(27.0, 33.4),
        lon_range=(113.0, 123.24)
    )
