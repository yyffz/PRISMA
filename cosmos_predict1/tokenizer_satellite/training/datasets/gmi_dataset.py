# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset class for loading GMI processed NPZ files for tokenizer training.

GMI数据格式：
- NPZ文件包含：
  - gmi_data: [nchannels, nlat, nlon] - 已归一化的GMI数据
  - observation_mask: [nlat, nlon] - 观测mask（1=有效，0=无效）
  - grid_lat, grid_lon: 网格坐标
  - 时间信息等元数据

输出格式：
- video: [C, T, H, W] tensor，其中T可以是1（单时刻）或num_video_frames（时间序列）
- observation_mask: [H, W] tensor（可选）
"""

import os
import warnings
import json
import hashlib
import pickle
from glob import glob
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


def _find_valid_crop_regions_worker(args):
    """
    并行处理的工作函数：查找单个样本的有效裁剪区域
    
    Args:
        args: 元组 (npz_file, crop_size, resize, lat_range, min_valid_observations)
    
    Returns:
        tuple: (cache_key, valid_regions) 或 (cache_key, None) 如果出错
    """
    npz_file, crop_size, resize, lat_range, min_valid_observations = args
    
    try:
        # 加载数据
        data = np.load(npz_file)
        
        # 使用gmi_data_raw进行归一化（与_load_npz_data和build_crop_cache.py保持一致）
        # 注意：必须使用gmi_data_raw，因为_load_npz_data在加载时使用gmi_data_raw进行归一化
        if 'gmi_data_raw' not in data:
            # 如果没有gmi_data_raw，回退到gmi_data
            gmi_data_raw = data.get('gmi_data', None)
            if gmi_data_raw is None:
                config_str = f"{crop_size}_{resize}_{lat_range}_{min_valid_observations}"
                key_str = f"{npz_file}_{config_str}"
                cache_key = hashlib.md5(key_str.encode()).hexdigest()
                return (cache_key, None)
            use_raw = False
        else:
            gmi_data_raw = data['gmi_data_raw']
            use_raw = True
        
        # 加载quality_s1和quality_s2，重新构造observation_mask（与_load_npz_data保持一致）
        quality_s1 = data.get('quality_s1', None)
        quality_s2 = data.get('quality_s2', None)
        
        # 重新构造observation_mask：只有当quality_s1和quality_s2都为0时，mask=1
        if quality_s1 is not None and quality_s2 is not None:
            # 只有当quality_s1 == 0 且 quality_s2 == 0 时，observation_mask = 1
            observation_mask = ((quality_s1 == 0) & (quality_s2 == 0)).astype(np.float32)
        else:
            # 如果没有quality数据，使用默认的observation_mask（如果存在）
            observation_mask = data.get('observation_mask', None)
            if observation_mask is not None:
                observation_mask = observation_mask.astype(np.float32)
        
        # 如果使用gmi_data_raw，需要加载统计文件进行归一化（与_load_npz_data保持一致）
        if use_raw:
            # 加载统计文件（延迟加载，只在第一次使用时加载）
            stats_path = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/yangyunfan/cosmos-predict1-main/datasets/GMI_processed/gmi_channel_stats.pth'
            if os.path.exists(stats_path):
                stats = torch.load(stats_path, map_location='cpu')
                min_quality_eq0 = stats['min_quality_eq0'].numpy() if isinstance(stats['min_quality_eq0'], torch.Tensor) else stats['min_quality_eq0']
                max_quality_eq0 = stats['max_quality_eq0'].numpy() if isinstance(stats['max_quality_eq0'], torch.Tensor) else stats['max_quality_eq0']
                # 调整形状为 [13, 1, 1] 用于广播
                min_quality_eq0 = min_quality_eq0.reshape(13, 1, 1)
                max_quality_eq0 = max_quality_eq0.reshape(13, 1, 1)
                
                # 使用min_quality_eq0和max_quality_eq0进行归一化（与_load_npz_data保持一致）
                gmi_data = (gmi_data_raw[:13, :, :] - min_quality_eq0) / (max_quality_eq0 - min_quality_eq0 + 1e-8) * 2.0 - 1.0
                
                # 将observation_mask=0处的值都设为-1（背景值）（与_load_npz_data保持一致）
                if observation_mask is not None:
                    mask_3d = np.broadcast_to(observation_mask[np.newaxis, :, :], gmi_data.shape)
                    gmi_data = np.where(mask_3d > 0.5, gmi_data, -1.0)
            else:
                # 如果没有统计文件，回退到使用gmi_data（如果存在）
                gmi_data = data.get('gmi_data', gmi_data_raw[:13, :, :])
                if gmi_data.shape[0] > 13:
                    gmi_data = gmi_data[:13, :, :]
        else:
            # 使用gmi_data（已归一化）
            gmi_data = gmi_data_raw[:13, :, :] if gmi_data_raw.shape[0] > 13 else gmi_data_raw
        
        # 应用纬度范围裁剪
        if lat_range is not None:
            min_lat, max_lat = lat_range
            grid_lat = data.get('grid_lat', None)
            
            if grid_lat is not None:
                lat_mask = (grid_lat >= min_lat) & (grid_lat <= max_lat)
                lat_indices = np.where(lat_mask)[0]
                if len(lat_indices) > 0:
                    gmi_data = gmi_data[:, lat_indices, :]
                    if observation_mask is not None:
                        observation_mask = observation_mask[lat_indices, :]
            else:
                nlat_total = gmi_data.shape[1]
                lat_values = np.linspace(-90, 90, nlat_total)
                lat_mask = (lat_values >= min_lat) & (lat_values <= max_lat)
                lat_indices = np.where(lat_mask)[0]
                if len(lat_indices) > 0:
                    gmi_data = gmi_data[:, lat_indices, :]
                    if observation_mask is not None:
                        observation_mask = observation_mask[lat_indices, :]
        
        # 处理NaN值
        if np.isnan(gmi_data).any():
            gmi_data = np.nan_to_num(gmi_data, nan=-1.0)
        
        # 转换为torch tensor并resize
        gmi_tensor = torch.from_numpy(gmi_data).float()
        if torch.isnan(gmi_tensor).any():
            gmi_tensor = torch.nan_to_num(gmi_tensor, nan=-1.0)
        
        if resize is not None:
            gmi_tensor = gmi_tensor.unsqueeze(0)
            gmi_tensor = F.interpolate(
                gmi_tensor, 
                size=resize, 
                mode='bilinear', 
                align_corners=False
            )
            gmi_tensor = gmi_tensor.squeeze(0)
            
            if observation_mask is not None:
                observation_mask = torch.from_numpy(observation_mask).float().unsqueeze(0).unsqueeze(0)
                observation_mask = F.interpolate(
                    observation_mask,
                    size=resize,
                    mode='nearest',
                )
                observation_mask = observation_mask.squeeze(0).squeeze(0)
        
        # 转换为numpy
        gmi_data_processed = gmi_tensor.numpy()
        if observation_mask is not None:
            observation_mask_processed = observation_mask.numpy() if isinstance(observation_mask, torch.Tensor) else observation_mask
        else:
            observation_mask_processed = None
        
        # 获取尺寸
        _, h, w = gmi_data_processed.shape
        crop_h, crop_w = crop_size
        
        # 使用向量化操作优化查找
        valid_regions = []
        
        if observation_mask_processed is not None:
            # 使用滑动窗口计算有效观测点数量（向量化）
            # 使用卷积或累积和来加速
            mask_2d = observation_mask_processed > 0.5
            # 计算每个位置的累积和
            cumsum = np.cumsum(np.cumsum(mask_2d.astype(int), axis=0), axis=1)
            # 填充边界
            cumsum_padded = np.pad(cumsum, ((1, 0), (1, 0)), mode='constant')
            
            for start_h in range(0, h - crop_h + 1):
                for start_w in range(0, w - crop_w + 1):
                    end_h = start_h + crop_h
                    end_w = start_w + crop_w
                    # 使用累积和快速计算区域内的有效点数
                    valid_count = (cumsum_padded[end_h, end_w] - 
                                 cumsum_padded[start_h, end_w] - 
                                 cumsum_padded[end_h, start_w] + 
                                 cumsum_padded[start_h, start_w])
                    
                    if valid_count >= min_valid_observations:
                        valid_regions.append((start_h, start_w))
        else:
            # 使用数据中非-1的点
            data_2d = gmi_data_processed[0, :, :]
            valid_mask = (data_2d != -1)
            cumsum = np.cumsum(np.cumsum(valid_mask.astype(int), axis=0), axis=1)
            cumsum_padded = np.pad(cumsum, ((1, 0), (1, 0)), mode='constant')
            
            for start_h in range(0, h - crop_h + 1):
                for start_w in range(0, w - crop_w + 1):
                    end_h = start_h + crop_h
                    end_w = start_w + crop_w
                    valid_count = (cumsum_padded[end_h, end_w] - 
                                 cumsum_padded[start_h, end_w] - 
                                 cumsum_padded[end_h, start_w] + 
                                 cumsum_padded[start_h, start_w])
                    
                    if valid_count >= min_valid_observations:
                        valid_regions.append((start_h, start_w))
        
        # 生成缓存键
        config_str = f"{crop_size}_{resize}_{lat_range}_{min_valid_observations}"
        key_str = f"{npz_file}_{config_str}"
        cache_key = hashlib.md5(key_str.encode()).hexdigest()
        
        return (cache_key, valid_regions)
    except Exception as e:
        # 返回错误信息
        config_str = f"{crop_size}_{resize}_{lat_range}_{min_valid_observations}"
        key_str = f"{npz_file}_{config_str}"
        cache_key = hashlib.md5(key_str.encode()).hexdigest()
        return (cache_key, None)


class GMIDataset(Dataset):
    def __init__(
        self,
        data_pattern,
        num_video_frames=1,
        sequence_interval=1,
        start_frame_interval=1,
        use_time_sequence=False,
        is_image_mode=False,
        crop_size=None,
        resize=None,
        lat_range=None,
        min_valid_observations=1000,
        stride=1,
        use_preprocessed=False,
        include_mask_channel=True,
        expand_channels=None,
    ):
        """Dataset class for loading GMI processed NPZ files.
        
        Args:
            data_pattern (str): Pattern to match NPZ files, e.g., "datasets/GMI_processed/**/*_GMI.npz"
            num_video_frames (int): Number of frames per sequence. 
                If use_time_sequence=False, this should be 1 (single timestep).
                If use_time_sequence=True, multiple consecutive files will be loaded.
            sequence_interval (int): Interval between sampled files in a sequence (when use_time_sequence=True)
            start_frame_interval (int): Interval between starting frames (when use_time_sequence=True)
            use_time_sequence (bool): If True, load multiple consecutive files to form a time sequence.
                Files are sorted by filename (which contains timestamp) to ensure temporal order.
                If False, each file is treated as a single timestep.
            is_image_mode (bool): If True, output [C, H, W] instead of [C, T, H, W]
            crop_size (tuple or int): If tuple (h, w), crop to this size. If int, crop to (crop_size, crop_size).
                If None, no cropping is applied. Cropping is done randomly for data augmentation.
                Note: If use_preprocessed=True, this parameter is ignored (data is already cropped).
            resize (tuple or int): If tuple (h, w), resize to this size. If int, resize to (resize, resize).
                If None, no resizing is applied. Resize is done using bilinear interpolation.
                Note: resize is applied before crop if both are specified.
                Note: If use_preprocessed=True, this parameter is ignored (data is already processed).
            lat_range (tuple or None): Latitude range to keep, e.g., (-65, 65) for 65°N-65°S.
                If None, no latitude cropping is applied. Cropping is done based on grid_lat in NPZ file.
                If grid_lat is not available, assumes standard IMERG grid (-90° to 90°, 1800 rows).
                Note: If use_preprocessed=True, this parameter is ignored (data is already cropped).
            min_valid_observations (int): Minimum number of valid observations required in the cropped region.
                If the number of valid observations (mask=1) is less than this threshold, a warning is issued.
                Default is 1000.
                Note: If use_preprocessed=True, this parameter is ignored (data is already validated).
            stride (int): Stride used when building the crop regions cache. This should match the stride
                used in build_crop_cache.py to ensure cache compatibility. Default is 1.
                Note: If use_preprocessed=True, this parameter is ignored (no cache needed).
            use_preprocessed (bool): If True, assume data is already preprocessed (normalized and cropped).
                In this mode, NPZ files should contain:
                - gmi_data: [C, H, W] normalized brightness temperature (13 channels)
                - observation_mask: [H, W] observation mask
                All preprocessing steps (normalization, cropping, lat_range filtering) will be skipped.
                Default is False.
            include_mask_channel (bool): If True, adds an observation mask channel as the last channel.
                When True: output is [14, T, H, W] (13 GMI channels + 1 mask channel)
                When False: output is [13, T, H, W] (only GMI channels, no mask channel)
                The observation_mask will still be returned separately in the data dict.
                Default is True.
        
        Returns dict with:
            - video: [C, T, H, W] tensor
            - If include_mask_channel=True: [14, T, H, W] (13 GMI channels + 1 mask channel)
            - If include_mask_channel=False: [13, T, H, W] (only GMI channels)
            - observation_mask: [H, W] tensor (if available, always returned separately)
            - video_name: Dict with file metadata
        """
        super().__init__()
        self.data_pattern = data_pattern
        self.num_video_frames = num_video_frames
        self.sequence_interval = sequence_interval
        self.start_frame_interval = start_frame_interval
        self.use_time_sequence = use_time_sequence
        self.is_image_mode = is_image_mode  # 如果True，输出[C, H, W]而不是[C, T, H, W]
        
        # 处理crop_size和resize参数
        if crop_size is not None:
            if isinstance(crop_size, int):
                self.crop_size = (crop_size, crop_size)
            else:
                self.crop_size = tuple(crop_size)
        else:
            self.crop_size = None
        
        if resize is not None:
            if isinstance(resize, int):
                self.resize = (resize, resize)
            else:
                self.resize = tuple(resize)
        else:
            self.resize = None
        
        # 处理纬度范围裁剪
        if lat_range is not None:
            # Hydra/OmegaConf会将元组转换为ListConfig，需要先转换为普通Python类型
            try:
                # 尝试转换为列表（支持list, tuple, omegaconf.ListConfig等）
                lat_list = list(lat_range)
                if len(lat_list) == 2:
                    # 确保是数值类型
                    lat_list = [float(x) for x in lat_list]
                    self.lat_range = tuple(sorted(lat_list))  # 确保是 [min_lat, max_lat]
                else:
                    raise ValueError(f"lat_range must have 2 elements, got {len(lat_list)} elements: {lat_range}")
            except (TypeError, ValueError) as e:
                raise ValueError(f"lat_range must be a tuple/list of 2 numeric elements, got {lat_range} (type: {type(lat_range)}). Error: {e}")
        else:
            self.lat_range = None
        
        # 最小有效观测点数量
        self.min_valid_observations = min_valid_observations
        
        # 裁剪步长（用于匹配缓存文件）
        self.stride = stride
        
        # 是否使用预处理后的数据
        self.use_preprocessed = use_preprocessed
        
        # 是否将mask作为额外通道
        self.include_mask_channel = include_mask_channel
        
        # 通道扩展：如果指定，将单通道数据扩展到指定的通道数（通过复制）
        # 例如：expand_channels=3 会将 [1, H, W] 扩展为 [3, H, W]
        self.expand_channels = expand_channels
        
        if self.use_preprocessed:
            print("使用预处理后的数据模式：跳过标准化、裁剪和缓存步骤")
        
        # 注意：实际通道数会在加载数据时确定（可能是GMI的13通道或AGRI的14通道）
        # 这里只显示mask通道的配置，实际通道数会在ChannelAdapter中处理
        if self.include_mask_channel:
            print(f"Mask通道模式：将添加1个mask通道（实际通道数由数据文件和target_channels决定）")
        else:
            print(f"无Mask通道模式：不添加mask通道（实际通道数由数据文件决定）")
        
        # 获取所有NPZ文件并排序（按文件名中的时间戳）
        self.npz_files = sorted(glob(str(data_pattern), recursive=True))
        # 统一转换为绝对路径，确保与 build_crop_cache.py 中使用的 npz_file 字符串一致，
        # 这样基于 (npz_file, config_str) 生成的缓存键才能完全匹配
        self.npz_files = [os.path.abspath(f) for f in self.npz_files]
        # 过滤掉以"."开头的隐藏文件
        self.npz_files = [f for f in self.npz_files if not os.path.basename(f).startswith('.')]
        
        print(f"{len(self.npz_files)} GMI NPZ files found")
        
        if len(self.npz_files) == 0:
            raise ValueError(f"No NPZ files found matching pattern: {data_pattern}")
        
        # 初始化样本
        if use_time_sequence:
            self.samples = self._init_time_sequence_samples()
        else:
            self.samples = self._init_single_frame_samples()
        
        print(f"{len(self.samples)} samples in total")
        self.wrong_number = 0
        # 记录失败的文件，避免重复尝试
        self._failed_files = set()
        # 重试次数限制
        self._max_retries = 10
        
        # 裁剪区域缓存：存储每个样本的有效裁剪区域
        # 格式：{sample_key: [(start_h, start_w), ...]}
        self.crop_regions_cache = {}
        
        # 如果使用预处理后的数据，跳过缓存构建
        if self.use_preprocessed:
            print("预处理模式：跳过裁剪区域缓存构建")
        # 如果启用了裁剪，构建有效裁剪区域缓存
        elif self.crop_size is not None and self.min_valid_observations > 0:
            # 检测是否在分布式训练环境中
            try:
                import torch.distributed as dist
                is_distributed = dist.is_available() and dist.is_initialized()
            except:
                is_distributed = False
            
            print(f"构建有效裁剪区域缓存（crop_size={self.crop_size}, min_valid={self.min_valid_observations}）...")
            # 先尝试从磁盘加载缓存
            cache_loaded = self._load_crop_regions_cache()
            if not cache_loaded:
                # 在分布式环境中，绝对不构建缓存（避免阻塞其他进程导致NCCL超时）
                if is_distributed:
                    warnings.warn(
                        "在分布式训练环境中，缓存文件不存在。"
                        "为避免阻塞训练和NCCL超时，将使用无缓存模式（随机裁剪+检查）。"
                        "建议在训练前单独运行脚本构建缓存以提高效率。"
                    )
                    # 不构建缓存，使用旧逻辑（随机裁剪+检查）
                    # 缓存字典保持为空，_load_npz_data 会使用旧逻辑
                    self.crop_regions_cache = {}  # 确保为空
                else:
                    # 非分布式环境，可以立即构建缓存
                    print("缓存不存在，开始构建...")
                    self._build_crop_regions_cache()
                    self._save_crop_regions_cache()
            else:
                # 缓存已加载，过滤掉那些在缓存中没有有效区域的样本
                if len(self.crop_regions_cache) > 0:
                    original_count = len(self.samples)
                    # 过滤样本：只保留那些在缓存中有有效区域的样本
                    valid_samples = []
                    for sample in self.samples:
                        npz_file = sample.get('npz_file', None)
                        if npz_file is not None:
                            cache_key = self._get_cache_key(npz_file)
                            if cache_key in self.crop_regions_cache:
                                valid_samples.append(sample)
                    
                    self.samples = valid_samples
                    filtered_count = original_count - len(self.samples)
                    if filtered_count > 0:
                        print(f"根据缓存过滤掉 {filtered_count} 个无效样本（在缓存中没有有效区域）")
                        print(f"剩余有效样本数: {len(self.samples)}")
    
    def _init_single_frame_samples(self):
        """初始化单时刻样本（每个文件一个样本）"""
        samples = []
        
        # 跳过逐文件验证以加快初始化速度（对于大量文件，逐个打开验证会非常慢）
        # 假设所有文件格式一致（由同一脚本生成），只在实际加载时检查
        # 如果需要验证，可以设置环境变量 GMI_VALIDATE_FILES=1
        validate_files = os.environ.get('GMI_VALIDATE_FILES', '0') == '1'
        
        if validate_files:
            print("正在验证所有NPZ文件（这可能需要一些时间）...")
            for npz_file in self.npz_files:
                try:
                    # 验证文件是否可读
                    data = np.load(npz_file)
                    if 'gmi_data' not in data and 'gmi_data_raw' not in data:
                        warnings.warn(f"File {npz_file} missing 'gmi_data'/'gmi_data_raw' key. Skipping.")
                        continue
                    
                    sample = {
                        'npz_file': npz_file,
                        'frame_ids': [0],  # 单时刻，只有一个"帧"
                    }
                    samples.append(sample)
                except Exception as e:
                    warnings.warn(f"Failed to load {npz_file}: {e}. Skipping.")
                    continue
        else:
            # 快速模式：不验证文件，直接添加所有文件
            # 无效文件会在 __getitem__ 中被捕获并跳过
            for npz_file in self.npz_files:
                sample = {
                    'npz_file': npz_file,
                    'frame_ids': [0],  # 单时刻，只有一个"帧"
                }
                samples.append(sample)
        
        return samples
    
    def _init_time_sequence_samples(self):
        """初始化时间序列样本（多个连续文件组成一个序列）"""
        samples = []
        
        for start_idx in range(0, len(self.npz_files), self.start_frame_interval):
            sample = {
                'npz_file': self.npz_files[start_idx],  # 起始文件
                'frame_ids': [],
            }
            
            curr_idx = start_idx
            while len(sample['frame_ids']) < self.num_video_frames:
                if curr_idx >= len(self.npz_files):
                    break
                sample['frame_ids'].append(curr_idx)
                curr_idx += self.sequence_interval
            
            # 确保有足够的帧
            if len(sample['frame_ids']) == self.num_video_frames:
                samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _get_cache_key(self, npz_file):
        """生成缓存键（基于文件路径和配置参数）"""
        # 使用文件路径和配置参数生成唯一键（与build_crop_cache.py保持一致）
        # 注意：build_crop_cache.py 中使用的是：
        #   config_str = f"{crop_size}_{resize}_{lat_range}_{min_valid_observations}_{stride}"
        # 其中 crop_size 为 int（例如 256），resize 为 int 或 None，
        # lat_range 为 tuple：(-65.0, 65.0)（float tuple，因为 argparse 中 type=float）。
        # 在 GMIDataset 中，crop_size/resize 通常被转换为 tuple，lat_range 可能为 int 或 float tuple，
        # 因此需要对 crop_size/resize 做规范化处理，同时确保 lat_range 格式与 build_crop_cache.py 一致。
        crop_size_repr = self.crop_size
        if isinstance(crop_size_repr, tuple):
            # 如果是正方形裁剪，如 (256, 256)，转换为单个 int 以匹配脚本中的用法
            if len(crop_size_repr) == 2 and crop_size_repr[0] == crop_size_repr[1]:
                crop_size_repr = crop_size_repr[0]
        
        resize_repr = self.resize
        if isinstance(resize_repr, tuple):
            if len(resize_repr) == 2 and resize_repr[0] == resize_repr[1]:
                resize_repr = resize_repr[0]
        
        # lat_range 格式处理：确保与 build_crop_cache.py 一致
        # 注意：实际构建缓存时使用的是 int tuple（例如 (-65, 65)），
        # 因为 argparse 的默认值 [-65, 65] 在转换为 tuple 时可能被表示为 int tuple。
        # 这里需要转换为 int tuple（如果值是整数）以匹配缓存中的格式。
        lat_range_repr = self.lat_range
        if lat_range_repr is not None:
            try:
                lat_min, lat_max = lat_range_repr
                # 如果值是整数，转换为 int tuple 以匹配缓存中的格式
                if lat_min == int(lat_min) and lat_max == int(lat_max):
                    lat_range_repr = (int(lat_min), int(lat_max))
                else:
                    # 如果不是整数，转换为 float tuple
                    lat_range_repr = (float(lat_min), float(lat_max))
            except (ValueError, TypeError):
                # 如果转换失败，保持原样
                pass
        
        config_str = f"{crop_size_repr}_{resize_repr}_{lat_range_repr}_{self.min_valid_observations}_{self.stride}"
        key_str = f"{npz_file}_{config_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_file_path(self):
        """获取缓存文件路径（与build_crop_cache.py保持一致）"""
        # 基于配置参数生成缓存文件名（包含stride参数）
        # build_crop_cache.py 中的实现为：
        #   config_str = f"crop_{crop_size}_resize_{resize}_lat_{lat_range}_minobs_{min_valid_observations}_stride_{stride}"
        # 这里同样需要对 crop_size / resize 做与 _get_cache_key 相同的规范化处理，
        # 并确保 lat_range 格式与 build_crop_cache.py 一致（使用 int 而不是 float）
        crop_size_repr = self.crop_size
        if isinstance(crop_size_repr, tuple):
            if len(crop_size_repr) == 2 and crop_size_repr[0] == crop_size_repr[1]:
                crop_size_repr = crop_size_repr[0]
        
        resize_repr = self.resize
        if isinstance(resize_repr, tuple):
            if len(resize_repr) == 2 and resize_repr[0] == resize_repr[1]:
                resize_repr = resize_repr[0]
        
        # lat_range 格式处理：确保与 build_crop_cache.py 一致
        # build_crop_cache.py 中 lat_range 可能是 int 类型，所以这里转换为 int（如果可能）
        # 这样可以确保哈希值一致（例如：(-65, 65) 而不是 (-65.0, 65.0)）
        lat_range_repr = self.lat_range
        if lat_range_repr is not None:
            try:
                # 尝试转换为 int（如果值是整数）
                lat_min, lat_max = lat_range_repr
                if lat_min == int(lat_min) and lat_max == int(lat_max):
                    lat_range_repr = (int(lat_min), int(lat_max))
            except (ValueError, TypeError):
                # 如果转换失败，保持原样
                pass
        
        config_str = (
            f"crop_{crop_size_repr}_resize_{resize_repr}_lat_{lat_range_repr}_"
            f"minobs_{self.min_valid_observations}_stride_{self.stride}"
        )
        cache_filename = f"gmi_crop_regions_cache_{hashlib.md5(config_str.encode()).hexdigest()[:8]}.pkl"
        
        # 尝试在数据目录或当前目录创建缓存
        if self.npz_files:
            # 使用第一个文件的目录作为缓存目录
            data_dir = os.path.dirname(self.npz_files[0])
            cache_dir = os.path.join(data_dir, '.crop_cache')
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, cache_filename)
        else:
            return cache_filename
    
    def _load_crop_regions_cache(self):
        """从磁盘加载缓存"""
        cache_file = self._get_cache_file_path()
        if os.path.exists(cache_file):
            try:
                print(f"正在从缓存文件加载: {cache_file}")
                with open(cache_file, 'rb') as f:
                    raw_cache = pickle.load(f)
                
                # 过滤掉valid_regions为空的样本（这些样本没有有效区域）
                self.crop_regions_cache = {
                    k: v for k, v in raw_cache.items() 
                    if len(v) > 0  # 只保留有有效区域的样本
                }
                
                filtered_count = len(raw_cache) - len(self.crop_regions_cache)
                if filtered_count > 0:
                    print(f"过滤掉 {filtered_count} 个没有有效区域的样本")
                
                print(f"成功加载缓存，包含 {len(self.crop_regions_cache)} 个有效样本的有效区域")
                return True
            except Exception as e:
                warnings.warn(f"加载缓存文件失败: {e}，将重新构建缓存")
                return False
        return False
    
    def _save_crop_regions_cache(self):
        """保存缓存到磁盘"""
        cache_file = self._get_cache_file_path()
        try:
            print(f"正在保存缓存到: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.crop_regions_cache, f)
            print(f"缓存已保存，包含 {len(self.crop_regions_cache)} 个样本的有效区域")
        except Exception as e:
            warnings.warn(f"保存缓存文件失败: {e}")
    
    def _find_valid_crop_regions(self, npz_file):
        """
        查找一个样本中所有符合要求的裁剪区域
        
        Args:
            npz_file: NPZ文件路径
        
        Returns:
            list: 有效裁剪区域列表，每个元素是 (start_h, start_w)
        """
        # 加载数据（不进行裁剪）
        data = np.load(npz_file)
        gmi_data = data['gmi_data']
        observation_mask = data.get('observation_mask', None)
        
        # 应用纬度范围裁剪（如果指定）
        if self.lat_range is not None:
            min_lat, max_lat = self.lat_range
            grid_lat = data.get('grid_lat', None)
            
            if grid_lat is not None:
                lat_mask = (grid_lat >= min_lat) & (grid_lat <= max_lat)
                lat_indices = np.where(lat_mask)[0]
                if len(lat_indices) > 0:
                    gmi_data = gmi_data[:, lat_indices, :]
                    if observation_mask is not None:
                        observation_mask = observation_mask[lat_indices, :]
            else:
                nlat_total = gmi_data.shape[1]
                lat_values = np.linspace(-90, 90, nlat_total)
                lat_mask = (lat_values >= min_lat) & (lat_values <= max_lat)
                lat_indices = np.where(lat_mask)[0]
                if len(lat_indices) > 0:
                    gmi_data = gmi_data[:, lat_indices, :]
                    if observation_mask is not None:
                        observation_mask = observation_mask[lat_indices, :]
        
        # 处理NaN值
        if np.isnan(gmi_data).any():
            gmi_data = np.nan_to_num(gmi_data, nan=-1.0)
        
        # 转换为torch tensor
        gmi_tensor = torch.from_numpy(gmi_data).float()
        if torch.isnan(gmi_tensor).any():
            gmi_tensor = torch.nan_to_num(gmi_tensor, nan=-1.0)
        
        # 应用resize（如果指定）
        if self.resize is not None:
            gmi_tensor = gmi_tensor.unsqueeze(0)  # [1, C, H, W]
            gmi_tensor = F.interpolate(
                gmi_tensor, 
                size=self.resize, 
                mode='bilinear', 
                align_corners=False
            )
            gmi_tensor = gmi_tensor.squeeze(0)  # [C, H, W]
            
            if observation_mask is not None:
                observation_mask = torch.from_numpy(observation_mask).float().unsqueeze(0).unsqueeze(0)
                observation_mask = F.interpolate(
                    observation_mask,
                    size=self.resize,
                    mode='nearest',
                )
                observation_mask = observation_mask.squeeze(0).squeeze(0)
        
        # 转换为numpy用于检查
        if isinstance(gmi_tensor, torch.Tensor):
            gmi_data_processed = gmi_tensor.numpy()
        else:
            gmi_data_processed = gmi_tensor
        
        if observation_mask is not None:
            if isinstance(observation_mask, torch.Tensor):
                observation_mask_processed = observation_mask.numpy()
            else:
                observation_mask_processed = observation_mask
        else:
            observation_mask_processed = None
        
        # 获取图像尺寸和裁剪尺寸
        _, h, w = gmi_data_processed.shape
        crop_h, crop_w = self.crop_size
        
        # 遍历所有可能的裁剪位置
        valid_regions = []
        
        for start_h in range(0, h - crop_h + 1):
            for start_w in range(0, w - crop_w + 1):
                # 提取裁剪区域
                crop_h_end = start_h + crop_h
                crop_w_end = start_w + crop_w
                
                # 检查有效观测点数量
                if observation_mask_processed is not None:
                    crop_mask = observation_mask_processed[start_h:crop_h_end, start_w:crop_w_end]
                    valid_count = int(np.sum(crop_mask > 0.5))
                else:
                    # 使用数据中非-1的点作为有效观测点
                    crop_data = gmi_data_processed[0, start_h:crop_h_end, start_w:crop_w_end]
                    valid_mask = (crop_data != -1)
                    valid_count = int(np.sum(valid_mask))
                
                # 如果有效观测点数量满足要求，添加到有效区域列表
                if valid_count >= self.min_valid_observations:
                    valid_regions.append((start_h, start_w))
        
        return valid_regions
    
    def _build_crop_regions_cache(self, num_workers=None):
        """
        为所有样本构建有效裁剪区域缓存（使用并行处理）
        
        Args:
            num_workers: 并行处理的进程数，如果为None则使用CPU核心数
        """
        # 检测是否在分布式训练环境中
        try:
            import torch.distributed as dist
            is_distributed = dist.is_available() and dist.is_initialized()
            is_rank0 = is_distributed and dist.get_rank() == 0
        except:
            is_distributed = False
            is_rank0 = True
        
        # 在分布式环境中，绝对不构建缓存（避免阻塞导致NCCL超时）
        # 即使 rank0 也不构建，因为构建时间太长会导致其他进程超时
        if is_distributed:
            if not is_rank0:
                print("非 rank0 进程，跳过缓存构建...")
                # 尝试加载缓存（如果存在）
                if self._load_crop_regions_cache():
                    return
                else:
                    warnings.warn(
                        "缓存不存在，非 rank0 进程将使用无缓存模式。"
                        "建议在训练前单独运行脚本构建缓存。"
                    )
                    # 返回空缓存，使用旧逻辑（随机裁剪+检查）
                    return
            else:
                # rank0 也不构建，避免长时间阻塞导致NCCL超时
                warnings.warn(
                    "在分布式训练环境中，rank0 也不构建缓存以避免NCCL超时。"
                    "建议在训练前单独运行脚本构建缓存。"
                )
                return
        
        print("正在扫描所有样本以查找有效裁剪区域（使用并行处理）...")
        
        # 在分布式环境中，避免使用 ProcessPoolExecutor（会导致 Bus error）
        # 改用 ThreadPoolExecutor 或单进程
        use_threads = is_distributed
        
        if num_workers is None:
            import multiprocessing
            if use_threads:
                # 使用线程池，避免 fork 问题
                num_workers = min(multiprocessing.cpu_count(), 8)  # 线程数可以多一些
            else:
                num_workers = min(multiprocessing.cpu_count(), 16)  # 进程数限制更严格
        
        total_samples = len(self.samples)
        valid_samples = 0
        invalid_samples = 0
        
        # 准备参数
        args_list = [
            (
                sample['npz_file'],
                self.crop_size,
                self.resize,
                self.lat_range,
                self.min_valid_observations
            )
            for sample in self.samples
        ]
        
        # 使用并行处理
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        executor_name = "线程" if use_threads else "进程"
        print(f"使用 {num_workers} 个{executor_name}并行处理...")
        
        with executor_class(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_sample = {
                executor.submit(_find_valid_crop_regions_worker, args): args[0]
                for args in args_list
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_sample), total=len(args_list), desc="构建裁剪区域缓存"):
                npz_file = future_to_sample[future]
                try:
                    cache_key, valid_regions = future.result()
                    
                    if valid_regions is not None:
                        if len(valid_regions) > 0:
                            self.crop_regions_cache[cache_key] = valid_regions
                            valid_samples += 1
                        else:
                            self.crop_regions_cache[cache_key] = []
                            invalid_samples += 1
                            if invalid_samples <= 10:
                                warnings.warn(
                                    f"样本 {os.path.basename(npz_file)} 没有找到符合要求的裁剪区域 "
                                    f"(min_valid={self.min_valid_observations})"
                                )
                    else:
                        # 处理出错的情况
                        self.crop_regions_cache[cache_key] = []
                        invalid_samples += 1
                        warnings.warn(f"处理样本 {os.path.basename(npz_file)} 时出错")
                except Exception as e:
                    invalid_samples += 1
                    warnings.warn(f"处理样本 {os.path.basename(npz_file)} 时出错: {e}")
        
        # 统计信息
        total_valid_regions = sum(len(regions) for regions in self.crop_regions_cache.values())
        avg_regions_per_sample = total_valid_regions / valid_samples if valid_samples > 0 else 0
        
        print(f"裁剪区域缓存构建完成:")
        print(f"  - 有效样本: {valid_samples}/{total_samples}")
        print(f"  - 无效样本: {invalid_samples}/{total_samples}")
        print(f"  - 总有效裁剪区域数: {total_valid_regions}")
        print(f"  - 平均每个样本的有效区域数: {avg_regions_per_sample:.1f}")
        
        if invalid_samples > 0:
            warnings.warn(
                f"有 {invalid_samples} 个样本没有找到有效裁剪区域，这些样本将被跳过"
            )
    
    def _load_preprocessed_data(self, data, npz_file):
        """
        加载预处理后的NPZ文件数据（已标准化和裁剪）
        
        支持多种格式：
        1. GMI格式：gmi_data [C, H, W] 或 [C, T, H, W]
        2. AGRI格式：agri_data [C, H, W] 或 [C, T, H, W]
        3. IMERG格式：imerg_data [1, H, W]（单通道降水数据）
        4. DPR Ka格式：dpr_ka_data [1, H, W]（单通道组合反射率数据）
        
        Args:
            data: 已加载的NPZ数据字典
            npz_file: NPZ文件路径（用于错误信息）
        
        Returns:
            tuple: (data_tensor, observation_mask)
                - data_tensor: [C, T, H, W] 或 [C, H, W] tensor
                - observation_mask: [T, H, W] 或 [H, W] tensor
        """
        # 检测数据格式（GMI、AGRI、IMERG或MESO）
        is_imerg = False
        is_meso = False
        is_dpr_ka = False
        if 'meso_data' in data:
            # CMA-MESO格式（多变量、可能多垂直层的气象场数据）
            raw_data = data['meso_data']  # [C, T, H, W] 或 [C, H, W]
            is_agri = False
            is_meso = True
            expected_channels = raw_data.shape[0]  # MESO通道数由数据决定
        elif 'imerg_data' in data:
            # IMERG格式（单通道降水数据）
            raw_data = data['imerg_data']  # [1, H, W]
            is_agri = False
            is_imerg = True
            expected_channels = 1  # IMERG只有1个通道
        elif 'dpr_ka_data' in data:
            # DPR Ka格式（单通道组合反射率数据）
            raw_data = data['dpr_ka_data']  # [1, H, W]
            is_agri = False
            is_dpr_ka = True
            expected_channels = 1
        elif 'agri_data' in data:
            # AGRI格式
            raw_data = data['agri_data']  # [C, T, H, W] 或 [C, H, W]
            is_agri = True
            expected_channels = 9  # AGRI有9个通道（中红外和热红外通道，如果包含mask则为10）
        elif 'gmi_data' in data:
            # GMI格式
            raw_data = data['gmi_data']  # [13, H, W] 或 [13, T, H, W]
            is_agri = False
            expected_channels = 13  # GMI有13个通道
        else:
            raise ValueError(
                f"预处理后的文件 {os.path.basename(npz_file)} 缺少 "
                "'gmi_data'、'agri_data'、'imerg_data'、'dpr_ka_data' 或 'meso_data' 键"
            )
        
        # 处理observation_mask（IMERG使用quality_index作为替代）
        if is_imerg:
            # IMERG: 优先使用quality_index，如果没有则回退到observation_mask
            observation_mask = data.get('quality_index', data.get('observation_mask', None))
        elif is_dpr_ka:
            # DPR Ka: 使用dpr_ka_mask标注有无观测，如果没有则回退到observation_mask
            observation_mask = data.get('dpr_ka_mask', data.get('observation_mask', None))
        else:
            observation_mask = data.get('observation_mask', None)
        
        # 检测数据维度（3D图像或4D视频）
        is_video = len(raw_data.shape) == 4
        
        if is_video:
            # 视频格式: [C, T, H, W]
            C, T, H, W = raw_data.shape
            if is_agri:
                # AGRI: 如果包含mask通道，应该是10个通道，否则9个（中红外和热红外通道）
                if C not in [9, 10]:
                    warnings.warn(
                        f"预处理后的 agri_data 应该有9或10个通道，但得到 {C} 个通道。"
                    )
            else:
                # GMI: 应该是13个通道
                if C != expected_channels:
                    warnings.warn(
                        f"预处理后的 gmi_data 应该有{expected_channels}个通道，但得到 {C} 个通道。"
                        f"将使用前{expected_channels}个通道。"
                    )
                    raw_data = raw_data[:expected_channels, :, :, :]
            
            # 转换为torch tensor
            data_tensor = torch.from_numpy(raw_data).float()  # [C, T, H, W]
            
            # 处理observation_mask（视频格式）
            if observation_mask is not None:
                observation_mask = torch.from_numpy(observation_mask).float()
                # 确保mask维度正确
                if len(observation_mask.shape) == 2:
                    # [H, W] -> [T, H, W] (广播到所有时间步)
                    observation_mask = observation_mask.unsqueeze(0).expand(T, -1, -1)
                elif len(observation_mask.shape) == 3:
                    # [T, H, W] 或 [C, H, W]
                    if observation_mask.shape[0] == T:
                        # [T, H, W]
                        pass
                    elif observation_mask.shape[0] == C:
                        # [C, H, W] -> 取第一个通道并广播到所有时间步
                        observation_mask = observation_mask[0:1, :, :].expand(T, -1, -1)
                    else:
                        # 假设是 [H, W]，广播到 [T, H, W]
                        observation_mask = observation_mask.unsqueeze(0).expand(T, -1, -1)
                # 确保mask值在[0, 1]范围内
                observation_mask = torch.clamp(observation_mask, 0.0, 1.0)
            else:
                # 如果没有mask，创建一个全1的mask（假设所有数据都有效）
                observation_mask = torch.ones(T, H, W, dtype=torch.float32)
                if not is_meso:  # MESO数据没有观测mask是正常的
                    warnings.warn(f"预处理后的文件 {os.path.basename(npz_file)} 缺少 'observation_mask'，使用全1掩码")
            
        else:
            # 图像格式: [C, H, W]
            C, H, W = raw_data.shape
            if is_agri:
                # AGRI: 如果包含mask通道，应该是10个通道，否则9个（中红外和热红外通道）
                if C not in [9, 10]:
                    warnings.warn(
                        f"预处理后的 agri_data 应该有9或10个通道，但得到 {C} 个通道。"
                    )
            else:
                # GMI: 应该是13个通道
                if C != expected_channels:
                    warnings.warn(
                        f"预处理后的 gmi_data 应该有{expected_channels}个通道，但得到 {C} 个通道。"
                        f"将使用前{expected_channels}个通道。"
                    )
                    raw_data = raw_data[:expected_channels, :, :]
            
            # 转换为torch tensor
            data_tensor = torch.from_numpy(raw_data).float()  # [C, H, W]
            
            # 处理observation_mask（图像格式）
            if observation_mask is not None:
                observation_mask = torch.from_numpy(observation_mask).float()  # [H, W] 或 [C, H, W]
                # 确保mask维度正确
                if len(observation_mask.shape) == 3:
                    # [C, H, W] -> 取第一个通道
                    observation_mask = observation_mask[0, :, :]
                elif len(observation_mask.shape) != 2:
                    raise ValueError(f"observation_mask 应该是 [H, W] 或 [C, H, W] 格式，但得到 {observation_mask.shape}")
                # 确保mask值在[0, 1]范围内
                observation_mask = torch.clamp(observation_mask, 0.0, 1.0)
            else:
                # 如果没有mask，创建一个全1的mask（假设所有数据都有效）
                observation_mask = torch.ones(H, W, dtype=torch.float32)
                if not is_meso:  # MESO数据没有观测mask是正常的
                    warnings.warn(f"预处理后的文件 {os.path.basename(npz_file)} 缺少 'observation_mask'，使用全1掩码")

        # 处理NaN值
        if torch.isnan(data_tensor).any():
            data_tensor = torch.nan_to_num(data_tensor, nan=-1.0)
        
        # 通道扩展：将单通道数据扩展到指定的通道数（通过复制）
        # 主要用于IMERG数据（1通道 -> 3通道）
        if self.expand_channels is not None and self.expand_channels > 1:
            current_channels = data_tensor.shape[0]
            if current_channels < self.expand_channels:
                # 计算需要复制的次数
                repeat_times = self.expand_channels // current_channels
                remainder = self.expand_channels % current_channels
                
                # 复制通道
                if is_video:
                    # [C, T, H, W] -> [expand_channels, T, H, W]
                    expanded = data_tensor.repeat(repeat_times, 1, 1, 1)
                    if remainder > 0:
                        expanded = torch.cat([expanded, data_tensor[:remainder, :, :, :]], dim=0)
                else:
                    # [C, H, W] -> [expand_channels, H, W]
                    expanded = data_tensor.repeat(repeat_times, 1, 1)
                    if remainder > 0:
                        expanded = torch.cat([expanded, data_tensor[:remainder, :, :]], dim=0)
                
                data_tensor = expanded

        # 根据include_mask_channel决定是否将mask作为最后一个通道添加到数据中。
        # 注意：必须先做expand_channels，再追加mask，DPR Ka需要 [reflectivity, reflectivity, mask]。
        if self.include_mask_channel:
            if is_video:
                mask_channel = observation_mask.unsqueeze(0)  # [1, T, H, W]
            else:
                mask_channel = observation_mask.unsqueeze(0)  # [1, H, W]
            data_tensor = torch.cat([data_tensor, mask_channel], dim=0)
        
        return data_tensor, observation_mask
    
    def _load_npz_data(self, npz_file, crop_position=None):
        """
        加载单个NPZ文件的数据
        
        Args:
            npz_file: NPZ文件路径
            crop_position: 裁剪位置 (start_h, start_w)，如果为None则随机选择（如果启用了缓存则从缓存中选择）
                Note: 如果use_preprocessed=True，此参数被忽略（数据已裁剪）
        """
        # 使用allow_pickle=False以提高安全性和加载速度
        try:
            data = np.load(npz_file, allow_pickle=False)
        except Exception as e:
            raise IOError(f"Failed to load NPZ file {npz_file}: {e}")
        
        # 如果使用预处理后的数据，直接加载并返回
        if self.use_preprocessed:
            return self._load_preprocessed_data(data, npz_file)
        
        # 加载统计文件（延迟加载，只在第一次使用时加载）
        if not hasattr(self, '_stats_loaded'):
            stats_path = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/yangyunfan/cosmos-predict1-main/datasets/GMI_processed/gmi_channel_stats.pth'
            if os.path.exists(stats_path):
                self._stats = torch.load(stats_path, map_location='cpu')
                # 获取min_quality_eq0和max_quality_eq0，转换为numpy
                self._min_quality_eq0 = self._stats['min_quality_eq0'].numpy() if isinstance(self._stats['min_quality_eq0'], torch.Tensor) else self._stats['min_quality_eq0']
                self._max_quality_eq0 = self._stats['max_quality_eq0'].numpy() if isinstance(self._stats['max_quality_eq0'], torch.Tensor) else self._stats['max_quality_eq0']
                # 调整形状为 [13, 1, 1] 用于广播
                self._min_quality_eq0 = self._min_quality_eq0.reshape(13, 1, 1)
                self._max_quality_eq0 = self._max_quality_eq0.reshape(13, 1, 1)
            else:
                raise FileNotFoundError(f"统计文件未找到: {stats_path}")
            self._stats_loaded = True
        
        # 加载gmi_data_raw [nchannels, nlat, nlon]
        gmi_data_raw = data['gmi_data_raw']
        
        # 加载quality_s1和quality_s2 [nlat, nlon]
        quality_s1 = data.get('quality_s1', None)
        quality_s2 = data.get('quality_s2', None)
        
        # 重新构造observation_mask：只有当quality_s1和quality_s2都为0时，mask=1
        if quality_s1 is not None and quality_s2 is not None:
            # 只有当quality_s1 == 0 且 quality_s2 == 0 时，observation_mask = 1
            observation_mask = ((quality_s1 == 0) & (quality_s2 == 0)).astype(np.float32)
        else:
            # 如果没有quality数据，使用默认的observation_mask（如果存在）
            observation_mask = data.get('observation_mask', None)
            if observation_mask is not None:
                observation_mask = observation_mask.astype(np.float32)
        
        # 使用min_quality_eq0和max_quality_eq0进行归一化
        # 归一化公式：normalized = (x - min) / (max - min) * 2 - 1，将值映射到[-1, 1]
        gmi_data = (gmi_data_raw[:13, :, :] - self._min_quality_eq0) / (self._max_quality_eq0 - self._min_quality_eq0 + 1e-8) * 2.0 - 1.0
        
        # 将observation_mask=0处的值都设为-1（背景值）
        if observation_mask is not None:
            # 将mask扩展到通道维度 [1, nlat, nlon] -> [13, nlat, nlon]
            mask_3d = np.broadcast_to(observation_mask[np.newaxis, :, :], gmi_data.shape)
            # 将mask=0处的值设为-1
            gmi_data = np.where(mask_3d > 0.5, gmi_data, -1.0)
        
        # 应用纬度范围裁剪（如果指定，在resize和crop之前）
        if self.lat_range is not None:
            min_lat, max_lat = self.lat_range
            grid_lat = data.get('grid_lat', None)
            
            if grid_lat is not None:
                # 使用NPZ文件中的grid_lat信息
                # grid_lat是一维数组，从南到北（-90°到90°）
                lat_mask = (grid_lat >= min_lat) & (grid_lat <= max_lat)
                lat_indices = np.where(lat_mask)[0]
                
                if len(lat_indices) == 0:
                    warnings.warn(
                        f"No data in latitude range [{min_lat}, {max_lat}] for {os.path.basename(npz_file)}. "
                        f"Available range: [{grid_lat.min():.2f}, {grid_lat.max():.2f}]"
                    )
                else:
                    # 裁剪数据：保留纬度范围内的行
                    gmi_data = gmi_data[:, lat_indices, :]  # [C, nlat_cropped, nlon]
                    if observation_mask is not None:
                        observation_mask = observation_mask[lat_indices, :]  # [nlat_cropped, nlon]
            else:
                # 如果没有grid_lat，假设是标准IMERG网格（-90°到90°，1800行）
                # 计算需要保留的行索引
                nlat_total = gmi_data.shape[1]  # 通常是1800
                # IMERG网格：第0行对应-90°，最后一行对应90°
                lat_values = np.linspace(-90, 90, nlat_total)
                lat_mask = (lat_values >= min_lat) & (lat_values <= max_lat)
                lat_indices = np.where(lat_mask)[0]
                
                if len(lat_indices) > 0:
                    gmi_data = gmi_data[:, lat_indices, :]  # [C, nlat_cropped, nlon]
                    if observation_mask is not None:
                        observation_mask = observation_mask[lat_indices, :]  # [nlat_cropped, nlon]
                else:
                    warnings.warn(
                        f"No data in latitude range [{min_lat}, {max_lat}] for {os.path.basename(npz_file)}. "
                        f"Assuming standard IMERG grid with {nlat_total} rows."
                    )
        
        # 处理NaN值：将NaN替换为-1（训练时不允许NaN值）
        # 这通常发生在插值过程中，当所有邻居点都超出距离阈值时
        if np.isnan(gmi_data).any():
            nan_count = np.isnan(gmi_data).sum()
            total_count = gmi_data.size
            nan_ratio = nan_count / total_count * 100
            warnings.warn(
                f"Found {nan_count} NaN values ({nan_ratio:.2f}%) in {os.path.basename(npz_file)}. "
                f"Replacing with -1."
            )
            gmi_data = np.nan_to_num(gmi_data, nan=-1.0)
        
        # 转换为torch tensor
        gmi_tensor = torch.from_numpy(gmi_data).float()  # [C, H, W]
        
        # 确保tensor中也没有NaN（双重保险）
        if torch.isnan(gmi_tensor).any():
            gmi_tensor = torch.nan_to_num(gmi_tensor, nan=-1.0)
        
        # 应用resize（如果指定）
        if self.resize is not None:
            # F.interpolate expects [B, C, H, W], so add batch dimension
            gmi_tensor = gmi_tensor.unsqueeze(0)  # [1, C, H, W]
            gmi_tensor = F.interpolate(
                gmi_tensor, 
                size=self.resize, 
                mode='bilinear', 
                align_corners=False
            )
            gmi_tensor = gmi_tensor.squeeze(0)  # [C, H, W]
            
            # 同样resize mask
            if observation_mask is not None:
                observation_mask = torch.from_numpy(observation_mask).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                observation_mask = F.interpolate(
                    observation_mask,
                    size=self.resize,
                    mode='nearest',  # mask使用nearest插值
                )
                observation_mask = observation_mask.squeeze(0).squeeze(0)  # [H, W]
        
        # 应用crop（如果指定，在resize之后）
        if self.crop_size is not None:
            _, h, w = gmi_tensor.shape
            crop_h, crop_w = self.crop_size
            
            # 确定裁剪位置
            if crop_position is not None:
                # 使用指定的裁剪位置
                start_h, start_w = crop_position
            elif self.min_valid_observations > 0 and len(self.crop_regions_cache) > 0:
                # print('从缓存的有效区域中随机选择')
                cache_key = self._get_cache_key(npz_file)
                valid_regions = self.crop_regions_cache.get(cache_key, [])
                
                if len(valid_regions) > 0:
                    # print('随机选择一个有效区域（使用numpy的随机数生成器）')
                    idx = np.random.randint(0, len(valid_regions))
                    start_h, start_w = valid_regions[idx]
                else:
                    print('如果没有有效区域（缓存中没有该样本或该样本没有有效区域），使用旧的随机裁剪逻辑作为降级处理')
                    # 如果没有有效区域（缓存中没有该样本或该样本没有有效区域），
                    # 使用旧的随机裁剪逻辑作为降级处理
                    # 注意：这可能会导致不满足min_valid_observations的情况，但会在后续检查中被捕获
                    if h >= crop_h and w >= crop_w:
                        start_h = torch.randint(0, h - crop_h + 1, size=(1,)).item()
                        start_w = torch.randint(0, w - crop_w + 1, size=(1,)).item()
                    else:
                        # 如果图像小于裁剪尺寸，从中心裁剪（降级处理）
                        start_h = max(0, (h - crop_h) // 2)
                        start_w = max(0, (w - crop_w) // 2)
                        crop_h = min(crop_h, h - start_h)
                        crop_w = min(crop_w, w - start_w)
            else:
                print('随机裁剪（数据增强）- 旧逻辑，用于没有启用缓存的情况')
                if h >= crop_h and w >= crop_w:
                    start_h = torch.randint(0, h - crop_h + 1, size=(1,)).item()
                    start_w = torch.randint(0, w - crop_w + 1, size=(1,)).item()
                else:
                    # 如果图像小于裁剪尺寸，从中心裁剪（降级处理）
                    start_h = max(0, (h - crop_h) // 2)
                    start_w = max(0, (w - crop_w) // 2)
                    crop_h = min(crop_h, h - start_h)
                    crop_w = min(crop_w, w - start_w)
            
            gmi_tensor = gmi_tensor[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            # 同样crop mask
            if observation_mask is not None:
                observation_mask = observation_mask[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        if observation_mask is not None and not isinstance(observation_mask, torch.Tensor):
            observation_mask = torch.from_numpy(observation_mask).float()  # [H, W]
        elif observation_mask is None:
            observation_mask = None
        
        # 检查有效观测点数量（在裁剪之后）
        # 注意：如果使用了缓存机制，这个检查通常已经通过（因为只选择有效区域）
        # 但为了安全起见，仍然进行检查
        if self.min_valid_observations > 0:
            if observation_mask is not None:
                # 使用observation_mask统计有效观测点（mask=1表示有效）
                if isinstance(observation_mask, torch.Tensor):
                    valid_count = int(torch.sum(observation_mask > 0.5).item())
                else:
                    valid_count = int(np.sum(observation_mask > 0.5))
                if valid_count < self.min_valid_observations:
                    raise ValueError(
                        f"Insufficient valid observations in cropped region of {os.path.basename(npz_file)}: "
                        f"found {valid_count} valid observations, required at least {self.min_valid_observations}. "
                        f"Latitude range: {self.lat_range if self.lat_range else 'full range'}. "
                        f"Sample will be skipped."
                    )
            else:
                # 如果没有observation_mask，使用数据中非-1的点作为有效观测点
                if isinstance(gmi_tensor, torch.Tensor):
                    valid_mask = (gmi_tensor[0, :, :] != -1)
                    valid_count = int(torch.sum(valid_mask).item())
                else:
                    valid_mask = (gmi_tensor[0, :, :] != -1)
                    valid_count = int(np.sum(valid_mask))
                if valid_count < self.min_valid_observations:
                    raise ValueError(
                        f"Insufficient valid observations in cropped region of {os.path.basename(npz_file)}: "
                        f"found {valid_count} valid observations (estimated from first channel), "
                        f"required at least {self.min_valid_observations}. "
                        f"Latitude range: {self.lat_range if self.lat_range else 'full range'}. "
                        f"Sample will be skipped."
                    )
        
        # 根据include_mask_channel决定是否将mask作为额外通道添加到数据中
        if self.include_mask_channel and observation_mask is not None:
            # 扩展mask维度: [H, W] -> [1, H, W]
            if isinstance(observation_mask, torch.Tensor):
                mask_channel = observation_mask.unsqueeze(0)  # [1, H, W]
            else:
                mask_channel = torch.from_numpy(observation_mask).float().unsqueeze(0)  # [1, H, W]
            
            # 拼接: [13, H, W] + [1, H, W] -> [14, H, W]
            gmi_tensor = torch.cat([gmi_tensor, mask_channel], dim=0)  # [14, H, W]
        # 如果include_mask_channel=False，gmi_tensor保持为[13, H, W]，observation_mask单独返回
        
        return gmi_tensor, observation_mask
    
    def __getitem__(self, index, retry_count=0):
        """
        获取数据样本
        
        Args:
            index: 样本索引
            retry_count: 重试次数（内部使用，避免无限递归）
        """
        # 限制重试次数，避免无限递归导致NCCL超时
        if retry_count >= self._max_retries:
            raise RuntimeError(
                f"Failed to load valid sample after {self._max_retries} retries. "
                f"Last attempted file: {self.samples[index].get('npz_file', 'unknown')}. "
                f"This may indicate data corruption or configuration issues."
            )
        
        try:
            sample = self.samples[index]
            npz_file = sample['npz_file']
            
            # 如果这个文件之前失败过，直接跳过
            if npz_file in self._failed_files:
                if len(self.samples) > 0:
                    # 随机选择另一个样本
                    new_index = np.random.randint(len(self.samples))
                    return self.__getitem__(new_index, retry_count + 1)
                else:
                    raise RuntimeError("No valid samples in dataset")
            
            frame_ids = sample['frame_ids']
            
            data = dict()
            
            if self.use_time_sequence:
                # 加载多个文件组成时间序列
                frames = []
                masks = []
                
                for frame_id in frame_ids:
                    npz_file_path = self.npz_files[frame_id]
                    # 检查文件是否在失败列表中
                    if npz_file_path in self._failed_files:
                        raise ValueError(f"File {npz_file_path} is in failed files list")
                    gmi_tensor, obs_mask = self._load_npz_data(npz_file_path)
                    frames.append(gmi_tensor)  # [C, H, W]
                    if obs_mask is not None:
                        masks.append(obs_mask)  # [H, W]
                
                # 堆叠成时间序列: [C, T, H, W]
                video = torch.stack(frames, dim=1)  # [C, T, H, W]
                
                if len(masks) > 0:
                    # 堆叠mask: [T, H, W]
                    observation_mask = torch.stack(masks, dim=0)  # [T, H, W]
                    # 如果所有时刻的mask相同，可以只保留一个: [H, W]
                    if observation_mask.shape[0] == 1:
                        observation_mask = observation_mask[0]  # [H, W]
                    data['observation_mask'] = observation_mask
                
                # 元数据：使用起始文件的信息
                base_name = os.path.basename(npz_file)
            else:
                # 单时刻数据（或已包含时间维度的视频数据）
                gmi_tensor, obs_mask = self._load_npz_data(npz_file)
                
                # 检测数据维度（3D图像或4D视频）
                is_video_data = len(gmi_tensor.shape) == 4
                
                if is_video_data:
                    # 数据已经是视频格式 [C, T, H, W]（例如AGRI预处理后的数据）
                    video = gmi_tensor  # [C, T, H, W]
                    if obs_mask is not None:
                        # obs_mask可能是 [T, H, W] 或 [H, W]
                        if len(obs_mask.shape) == 2:
                            # [H, W] -> 广播到 [T, H, W]
                            T = video.shape[1]
                            obs_mask = obs_mask.unsqueeze(0).expand(T, -1, -1)
                        data['observation_mask'] = obs_mask  # [T, H, W]
                elif self.is_image_mode:
                    # 图像模式：直接使用 [C, H, W]，不添加时间维度
                    video = gmi_tensor  # [C, H, W]
                    if obs_mask is not None:
                        data['observation_mask'] = obs_mask  # [H, W]
                else:
                    # 视频模式：添加时间维度 [C, 1, H, W]
                    video = gmi_tensor.unsqueeze(1)  # [C, T=1, H, W]
                if obs_mask is not None:
                    data['observation_mask'] = obs_mask  # [H, W]
                
                base_name = os.path.basename(npz_file)
            
            data['video'] = video
            # print(f'gmi_dataset data["video"].shape: {data["video"].shape}')
            
            data['video_name'] = {
                'npz_file': npz_file,
                'base_name': base_name,
            }
            
            # 获取数据形状信息
            # 直接根据video的实际维度判断，而不是依赖配置参数
            if len(video.shape) == 3:
                # 图像模式：[C, H, W]
                _, h, w = video.shape
            elif len(video.shape) == 4:
                # 视频模式：[C, T, H, W]
                _, _, h, w = video.shape
            else:
                raise ValueError(f"Unexpected video shape: {video.shape}, expected 3D [C, H, W] or 4D [C, T, H, W]")
            data['fps'] = 1  # GMI数据通常是每30分钟一个时刻
            data['image_size'] = torch.tensor([h, w, h, w])
            data['num_frames'] = self.num_video_frames
            data['padding_mask'] = torch.zeros(1, h, w)
            
            return data
            
        except Exception as e:
            # 记录失败的文件
            failed_file = self.samples[index].get('npz_file', 'unknown')
            self._failed_files.add(failed_file)
            self.wrong_number += 1
            
            # 只在重试次数较少时打印详细错误信息，避免日志过多
            if retry_count < 3:
                warnings.warn(
                    f"Invalid data encountered (retry {retry_count + 1}/{self._max_retries}): {failed_file}. "
                    f"Error: {e}. Trying another sample..."
                )
                if retry_count == 0:
                    import traceback
                    warnings.warn("FULL TRACEBACK:")
                    warnings.warn(traceback.format_exc())
            
            # 如果失败次数过多，打印警告
            if self.wrong_number % 100 == 0:
                print(f"Warning: {self.wrong_number} failed samples encountered. "
                      f"{len(self._failed_files)} unique files failed.")
            
            # 尝试加载另一个样本
            if len(self.samples) > 0:
                # 随机选择另一个样本
                new_index = np.random.randint(len(self.samples))
                return self.__getitem__(new_index, retry_count + 1)
            else:
                raise RuntimeError("No valid samples in dataset")


if __name__ == "__main__":
    # 测试代码
    dataset = GMIDataset(
        data_pattern="datasets/GMI_processed/**/*_GMI.npz",
        num_video_frames=1,
        use_time_sequence=False,
    )
    
    if len(dataset) > 0:
        print(f"Total samples: {len(dataset)}")
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Video shape: {sample['video'].shape}")
        if 'observation_mask' in sample:
            print(f"Observation mask shape: {sample['observation_mask'].shape}")
        print("Sample loaded successfully!")
