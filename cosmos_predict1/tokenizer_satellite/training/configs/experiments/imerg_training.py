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

"""Config settings for IMERG tokenizer training

IMERG数据是单通道降水数据（1通道），在dataloader中会被扩展为3通道以匹配预训练模型。
IMERG被视为图像数据（单帧），而非视频数据。

Example usage:
    # 使用QI通道（默认，4通道：3个扩展通道 + 1个QI通道）
    python train.py experiment=imerg_training
    
    # 不使用QI通道（3通道：仅扩展后的降水通道）
    python train.py experiment=imerg_training \\
        model.config.network.in_channels=3 \\
        model.config.network.out_channels=3 \\
        dataloader_train.dataset.target_channels=3 \\
        dataloader_train.dataset.include_mask_channel=False \\
        dataloader_val.dataset.target_channels=3 \\
        dataloader_val.dataset.include_mask_channel=False \\
        checkpoint.jit.input_shape=[1,3,256,256]
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict1.utils.lazy_config import LazyDict

# IMERG数据训练配置
# 是否使用QI（质量指数）通道：True=4通道（3个扩展通道+1个QI通道），False=3通道（仅扩展通道）
# 默认值：True（使用QI通道作为额外信息）
USE_QI_CHANNEL = True

# 通道配置
# IMERG原始数据是1通道，通过expand_channels扩展到3通道
# 如果使用QI通道，则最终为4通道（3+1）
EXPAND_CHANNELS = 3  # 将1通道扩展为3通道
NUM_CHANNELS = 4 if USE_QI_CHANNEL else 3

IMERG_TRAINING: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_image"},  # 使用图像tokenizer
            {"override /data_train": "gmi_loader_basic"},  # 复用GMI数据加载器
            {"override /data_val": "gmi_loader_basic"},
            {"override /loss": "video"},  # loss是通用的
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["basic"]},
            "_self_",
        ],
        # 模型配置
        model=dict(
            config=dict(
                network=dict(
                    # 通道数：3通道（扩展后）+ 1个QI通道（可选）= 4或3
                    in_channels=NUM_CHANNELS,
                    out_channels=NUM_CHANNELS,
                    # 空间压缩8倍（与预训练checkpoint一致）
                    spatial_compression=8,
                    # 使用patch_size=2（与CI8x8-360p配置一致）
                    patch_size=2,
                ),
                loss=dict(
                    config=dict(
                        # ColorLoss (L1): 主要的像素级重建损失
                        color=dict(
                            config=dict(
                                norm="L1",
                                boundaries=[0],
                                values=[1.5],  # 增强重建质量
                            )
                        ),
                        # KLLoss: 正则化潜在分布
                        kl=dict(
                            config=dict(
                                boundaries=[0],
                                values=[1e-6],
                            )
                        ),
                        # PerceptualLoss: 禁用，不适合降水数据
                        perceptual=dict(
                            config=dict(
                                lpips_boundaries=[0],
                                lpips_values=[0.0],
                                gram_enabled=False,
                                gram_boundaries=[0],
                                gram_values=[0.0],
                                corr_enabled=False,
                                corr_boundaries=[0],
                                corr_values=[0.0],
                                vgg_weights_path="/public/home/sunhaofei/cosmos-predict1/checkpoints/hub/checkpoints/vgg16-397923af.pth",
                            )
                        ),
                        # VideoConsistencyLoss: 禁用，图像数据不需要
                        video_consistency=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[0.0],
                                num_frames=1,
                                step=1,
                            )
                        ),
                        # FlowLoss: 禁用，图像数据不需要
                        flow=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[0.0],
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
            dataset=dict(
                dataset_name="gmi_processed",  # 复用GMI数据集类
                # IMERG预处理后的数据路径
                data_pattern="/public/share/users/sunhaofei/imerg_processed/*_imerg_*.npz",
                # 单时刻数据
                num_video_frames=1,
                use_time_sequence=False,
                # 通道配置
                target_channels=NUM_CHANNELS,
                channel_padding_mode="zero",
                # 通道扩展：将1通道扩展为3通道
                expand_channels=EXPAND_CHANNELS,
                # QI通道配置：如果启用，会将quality_index作为额外通道添加
                include_mask_channel=USE_QI_CHANNEL,
                # 使用预处理后的数据
                use_preprocessed=True,
                # 以下参数在use_preprocessed=True时会被忽略
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
        # 验证数据加载器配置
        dataloader_val=dict(
            dataset=dict(
                dataset_name="gmi_processed",
                data_pattern="/public/share/users/sunhaofei/imerg_processed/*_imerg_*.npz",
                num_video_frames=1,
                use_time_sequence=False,
                target_channels=NUM_CHANNELS,
                channel_padding_mode="zero",
                expand_channels=EXPAND_CHANNELS,
                include_mask_channel=USE_QI_CHANNEL,
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
        # 优化器配置：微调学习率
        optimizer=dict(
            lr=5e-5,  # 微调学习率
            betas=(0.5, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        ),
        # 学习率调度器配置
        scheduler=dict(
            warmup=10000,  # 更长的warmup有助于稳定微调
        ),
        job=dict(
            project="satellite_tokenizer",
            group="imerg",
            name="imerg_training_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        checkpoint=dict(
            # 使用预训练的Cosmos-Tokenize1-CI8x8-360p checkpoint
            load_path="/public/home/sunhaofei/cosmos-predict1/checkpoints/Cosmos-Tokenize1-CI8x8-360p/model.pt",
            # 图像tokenizer输入形状: [B, C, H, W]
            jit=dict(input_shape=[1, NUM_CHANNELS, 256, 256]),
            # 通道初始化策略
            channel_init_strategy="all_pretrained",
        ),
    )
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="imerg_training",
    node=IMERG_TRAINING
)
