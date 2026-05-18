#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立的 GMI 裁剪区域缓存构建脚本

在分布式训练前运行此脚本，预先构建缓存，避免训练时阻塞。

用法:
    python build_gmi_crop_cache.py \
        --data_pattern "datasets/GMI_processed/**/*_GMI.npz" \
        --crop_size 256 \
        --resize 256 \
        --lat_range -65 65 \
        --min_valid_observations 1000 \
        --num_workers 16
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from cosmos_predict1.tokenizer_satellite.training.datasets.gmi_dataset import GMIDataset


def main():
    parser = argparse.ArgumentParser(description='构建 GMI 裁剪区域缓存')
    parser.add_argument('--data_pattern', type=str, required=True,
                        help='数据文件模式，例如: datasets/GMI_processed/**/*_GMI.npz')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='裁剪尺寸（默认: 256）')
    parser.add_argument('--resize', type=int, default=None,
                        help='resize 尺寸（可选）')
    parser.add_argument('--lat_range', type=float, nargs=2, default=None,
                        help='纬度范围，例如: -65 65')
    parser.add_argument('--min_valid_observations', type=int, default=1000,
                        help='最小有效观测点数（默认: 1000）')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='并行处理的进程数（默认: CPU核心数）')
    
    args = parser.parse_args()
    
    # 创建数据集实例（这会自动构建缓存）
    print("=" * 80)
    print("GMI 裁剪区域缓存构建工具")
    print("=" * 80)
    print(f"数据模式: {args.data_pattern}")
    print(f"裁剪尺寸: {args.crop_size}")
    print(f"Resize: {args.resize}")
    print(f"纬度范围: {args.lat_range}")
    print(f"最小有效观测点: {args.min_valid_observations}")
    print(f"并行进程数: {args.num_workers}")
    print("=" * 80)
    
    try:
        dataset = GMIDataset(
            data_pattern=args.data_pattern,
            num_video_frames=1,
            use_time_sequence=False,
            is_image_mode=True,
            crop_size=args.crop_size,
            resize=args.resize,
            lat_range=tuple(args.lat_range) if args.lat_range else None,
            min_valid_observations=args.min_valid_observations,
        )
        
        # 如果缓存不存在，手动触发构建
        if len(dataset.crop_regions_cache) == 0:
            print("\n缓存不存在，开始构建...")
            dataset._build_crop_regions_cache(num_workers=args.num_workers)
            dataset._save_crop_regions_cache()
        else:
            print(f"\n缓存已存在，包含 {len(dataset.crop_regions_cache)} 个样本")
        
        print("\n" + "=" * 80)
        print("缓存构建完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

