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

"""Config settings for AGRI Image tokenizer training (Single Moment)

This configuration is specifically designed for training tokenizer on AGRI single-moment image NPZ files.
AGRI image data has shape (10, 256, 256): 10 channels (9 AGRI channels + 1 mask channel).
注意：这是针对单个时刻的AGRI图像训练，不是视频训练。

NPZ文件结构：
- agri_data: 形状 (C, H, W)，C=9个AGRI通道，H=W=256
- observation_mask: 形状 (H, W)，观测有效性mask
- h_start, w_start: 裁剪起始位置
- original_shape: 原始形状
- crop_size: 裁剪大小
- grid_lat, grid_lon: 经纬度网格

Example usage:
    # 使用mask通道（默认，10通道：9个AGRI通道 + 1个mask通道）
    python train.py experiment=agri_image_training \
        dataloader_train.dataset.data_pattern="/public/share/users/sunhaofei/yyf_data/AGRI_processed/preprocessed_image/*_agri_image_*.npz"
    
    # 不使用mask通道（9通道：仅AGRI通道）
    # 需要同时覆盖所有相关的通道数配置
    python train.py experiment=agri_image_training \
        model.config.network.in_channels=9 \
        model.config.network.out_channels=9 \
        dataloader_train.dataset.target_channels=9 \
        dataloader_train.dataset.include_mask_channel=False \
        dataloader_val.dataset.target_channels=9 \
        dataloader_val.dataset.include_mask_channel=False \
        checkpoint.jit.input_shape=[1,9,256,256] \
        dataloader_train.dataset.data_pattern="/public/share/users/sunhaofei/yyf_data/AGRI_processed/preprocessed_image/*_agri_image_*.npz"
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict1.utils.lazy_config import LazyDict

# AGRI图像数据训练配置
# 是否使用mask通道：True=10通道（9个AGRI通道+1个mask通道），False=9通道（仅AGRI通道）
# 默认值：True（使用mask通道）
# 注意：如果覆盖use_mask_channel，需要同时覆盖以下参数：
#   - model.config.network.in_channels (10或9)
#   - model.config.network.out_channels (10或9)
#   - dataloader_train.dataset.target_channels (10或9)
#   - dataloader_train.dataset.include_mask_channel (True或False)
#   - dataloader_val.dataset.target_channels (10或9)
#   - dataloader_val.dataset.include_mask_channel (True或False)
#   - checkpoint.jit.input_shape[1] (10或9)
USE_MASK_CHANNEL = True

# 根据是否使用mask通道确定通道数
# AGRI通道数：9个（中红外和热红外通道）
NUM_CHANNELS = 10 if USE_MASK_CHANNEL else 9

AGRI_IMAGE_TRAINING: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_image"},  # 使用图像tokenizer（单时刻）
            {"override /data_train": "gmi_loader_basic"},  # 复用GMI数据加载器（支持NPZ格式）
            {"override /data_val": "gmi_loader_basic"},  # 复用GMI数据加载器（支持NPZ格式）
            {"override /loss": "video"},  # loss是通用的，图像和视频都可以用
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["basic"]},
            "_self_",
        ],
        # 模型配置：根据use_mask_channel决定通道数
        model=dict(
            config=dict(
                network=dict(
                    # 通道数：如果使用mask通道则为10（9个AGRI通道+1个mask通道），否则为9（仅AGRI通道）
                    # 可以通过命令行覆盖：model.config.network.in_channels=9
                    in_channels=NUM_CHANNELS,
                    out_channels=NUM_CHANNELS,
                    # 空间压缩8倍（与CI8x8-360p配置一致）
                    spatial_compression=8,
                    # 使用patch_size=2（与CI8x8-360p配置一致）
                    patch_size=2,
                ),
                loss=dict(
                    config=dict(
                        # ColorLoss (L1): 主要的像素级重建损失，适合卫星数据
                        # 提高权重以确保良好的重建质量
                        color=dict(
                            config=dict(
                                norm="L1",
                                boundaries=[0],
                                values=[1.5],  # 从1.0提高到1.5，增强重建质量
                            )
                        ),
                        # KLLoss: 正则化潜在分布，权重很小，保持启用
                        kl=dict(
                            config=dict(
                                boundaries=[0],
                                values=[1e-6],  # 保持很小的权重
                            )
                        ),
                        # PerceptualLoss (LPIPS): 基于自然图像预训练的VGG16，不适合卫星数据
                        # 禁用或设置为0权重
                        perceptual=dict(
                            config=dict(
                                lpips_boundaries=[0],
                                lpips_values=[0.0],  # 设置为0，禁用LPIPS损失
                                gram_enabled=False,
                                gram_boundaries=[0],
                                gram_values=[0.0],  # 禁用Gram损失
                                corr_enabled=False,
                                corr_boundaries=[0],
                                corr_values=[0.0],  # 禁用Correlation损失
                                # 指定VGG16权重文件路径，避免在分布式训练时出现网络冲突
                                # 即使perceptual loss被禁用，也指定路径以确保初始化时不会触发下载
                                vgg_weights_path="/public/home/sunhaofei/cosmos-predict1/checkpoints/hub/checkpoints/vgg16-397923af.pth",
                            )
                        ),
                        # VideoConsistencyLoss: 不适用于单时刻图像数据，保持禁用
                        video_consistency=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[1.0],
                                num_frames=32,
                                step=8,
                            )
                        ),
                        # FlowLoss: 不适用于单时刻图像数据，保持禁用
                        flow=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[1_000_000],
                                values=[0.0, 0.01],
                                scale=2,
                                dtype="bfloat16",
                                checkpoint_activations=False,
                            )
                        ),
                    )
                )
            )
        ),
        # 训练数据加载器配置
        dataloader_train=dict(
            # 使用gmi_loader_basic（支持NPZ格式）
            dataset=dict(
                dataset_name="gmi_processed",  # 复用GMI数据集类（支持NPZ格式）
                # 数据路径模式（支持glob模式，包括**递归搜索）
                # 使用预处理后的图像数据：已标准化和裁剪的样本
                data_pattern="/public/share/users/sunhaofei/yyf_data/AGRI_processed/preprocessed_image/*_agri_image_*.npz",
                # 单时刻数据（每个NPZ文件是一个时刻）
                num_video_frames=1,
                use_time_sequence=False,
                # 通道配置：根据use_mask_channel决定通道数
                target_channels=NUM_CHANNELS,  # 如果use_mask_channel=True则为10（9个AGRI通道+1个mask通道），否则为9（仅AGRI通道）
                channel_padding_mode="zero",  # 可选: "zero", "repeat", "mean"
                # 观测mask配置：如果启用，会在数据通道后添加一个mask通道
                # mask通道表示有效观测区域（1=有效，0=无效）
                include_mask_channel=USE_MASK_CHANNEL,  # 设置为True以启用观测mask通道，False则不使用mask通道
                # 使用预处理后的数据：跳过标准化、裁剪和缓存步骤
                use_preprocessed=True,  # 设置为True以使用预处理后的数据
                # 以下参数在use_preprocessed=True时会被忽略，但保留用于兼容性
                lat_range=None,  # AGRI数据已经裁剪到256x256，不需要纬度范围过滤
                # 确保截取到的区域至少有足够的有效观测点，否则跳过该样本
                min_valid_observations=30000,  # 最少需要30000个有效观测点（mask=1），与预处理时保持一致
                crop_size=256,  # 从中心裁剪到256x256（预处理后数据已裁剪，此参数被忽略）
                stride=200,  # 与预处理时的stride一致（预处理后数据不需要缓存，此参数被忽略）
            ),
            batch_size=8,  # 图像数据可以使用更大的batch size
            num_workers=8,  # 数据加载worker数量
            prefetch_factor=4,  # 预取因子
            persistent_workers=False,  # 根据诊断结果，非持久worker性能更好
        ),
        # 验证数据加载器配置
        dataloader_val=dict(
            dataset=dict(
                dataset_name="gmi_processed",  # 复用GMI数据集类（支持NPZ格式）
                # 使用预处理后的数据：与训练保持一致
                data_pattern="/public/share/users/sunhaofei/yyf_data/AGRI_processed/preprocessed_image/*_agri_image_*.npz",
                num_video_frames=1,
                use_time_sequence=False,
                target_channels=NUM_CHANNELS,  # 与训练保持一致
                channel_padding_mode="zero",
                include_mask_channel=USE_MASK_CHANNEL,  # 与训练保持一致
                # 使用预处理后的数据：与训练保持一致
                use_preprocessed=True,
                lat_range=None,
                min_valid_observations=30000,
                crop_size=256,
                stride=200,
            ),
            batch_size=8,
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=False,
        ),
        # 优化器配置：针对微调任务调整学习率
        optimizer=dict(
            # 微调学习率：降低到预训练的 1/2，避免破坏预训练权重
            # 预训练使用 1e-4，微调使用 5e-5
            lr=5e-5,
            # 其他参数保持与预训练一致
            betas=(0.5, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        ),
        # 学习率调度器配置：增加 warmup 步数以稳定训练
        scheduler=dict(
            # 增加 warmup 步数：从 5000 增加到 10000
            # 更长的 warmup 有助于微调时稳定训练
            warmup=10000,
        ),
        job=dict(
            project="satellite_tokenizer",
            group="agri_image",
            name="agri_image_training_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        checkpoint=dict(
            # 使用预训练的Cosmos-Tokenize1-CI8x8-360p checkpoint（图像tokenizer）
            # 预训练权重为3通道RGB，将自适应扩展到10通道（9个AGRI通道+1个mask通道）
            load_path="/public/home/sunhaofei/cosmos-predict1/checkpoints/Cosmos-Tokenize1-CI8x8-360p/model.pt",
            # 图像tokenizer输入形状: [B, C, H, W]
            # 注意：图像tokenizer不需要时间维度T，C根据use_mask_channel决定（10或9）
            jit=dict(input_shape=[1, NUM_CHANNELS, 256, 256]),  # [B, C, H, W]，C=10（含mask）或9（不含mask），H=W=256
            # 通道初始化策略：当输入/输出通道数增加时的初始化方式
            # "first_only": 只有前N个通道（N=预训练通道数）使用预训练权重，新增通道使用随机初始化
            # "all_pretrained": 所有通道都使用预训练权重（新增通道循环复制预训练权重）
            channel_init_strategy="all_pretrained",  # 可选: "first_only" 或 "all_pretrained"
        ),
    )
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="agri_image_training",
    node=AGRI_IMAGE_TRAINING
)
