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

"""Config settings for AGRI tokenizer training

This configuration is specifically designed for training tokenizer on AGRI processed NPZ files.
AGRI data is video-like data with shape (10, 7, 256, 256): 10 channels (9 AGRI channels + 1 mask channel), 7 time steps.
注意：AGRI通道数已从15个改为9个（只使用中红外和热红外通道，即通道6-14）。

Example usage:
    # 使用mask通道（默认，10通道：9个AGRI通道 + 1个mask通道）
    python train.py experiment=agri_training \
        dataloader_train.dataset.data_pattern="/public/share/users/sunhaofei/AGRI_processed/preprocessed/*_agri_video_*.npz"
    
    # 不使用mask通道（9通道：仅AGRI通道）
    # 需要同时覆盖所有相关的通道数配置
    python train.py experiment=agri_training \
        model.config.network.in_channels=9 \
        model.config.network.out_channels=9 \
        dataloader_train.dataset.target_channels=9 \
        dataloader_train.dataset.include_mask_channel=False \
        dataloader_val.dataset.target_channels=9 \
        dataloader_val.dataset.include_mask_channel=False \
        checkpoint.jit.input_shape=[1,9,7,256,256] \
        dataloader_train.dataset.data_pattern="/public/share/users/sunhaofei/AGRI_processed/preprocessed/*_agri_video_*.npz"
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict1.utils.lazy_config import LazyDict

# AGRI数据训练配置
# 是否使用mask通道：True=10通道（9个AGRI通道+1个mask通道），False=9通道（仅AGRI通道）
# 注意：AGRI通道数已从15个改为9个（只使用中红外和热红外通道，即通道6-14）
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

# AGRI视频数据：7个时刻（用于VideoConsistencyLoss，num_frames=5，step=2产生2个窗口）
# 支持 frame_consistency 和 latent_consistency
NUM_TIME_STEPS = 7

AGRI_TRAINING: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_factorized_video"},  # 使用视频tokenizer（因子化版本）
            {"override /data_train": "gmi_loader_basic"},  # 复用GMI数据加载器（支持NPZ格式）
            {"override /data_val": "gmi_loader_basic"},  # 复用GMI数据加载器（支持NPZ格式）
            {"override /loss": "video"},  # 视频损失
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
                    # 时间压缩4倍（与CV4x8x8-360p配置一致）
                    temporal_compression=4,
                    # 空间压缩8倍（与CV4x8x8-360p配置一致）
                    spatial_compression=8,
                    # 使用patch_size=2（与CV4x8x8-360p配置一致）
                    patch_size=2,
                    # AGRI 为 9/10 通道输入，原始 16 维 latent 容量偏小；
                    # 扩展到 32 维，训练时会从 16 维预训练权重循环复制初始化。
                    latent_channels=32,
                    z_channels=32,
                ),
                channel_init_strategy="all_pretrained",
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
                        # VideoConsistencyLoss: 适用于视频数据
                        # 微调任务：5k步后启用（微调步数较少，需提前启用）
                        # 从头训练：建议改为250k步后启用
                        video_consistency=dict(
                            config=dict(
                                enabled=True,  # 启用视频一致性损失
                                boundaries=[5_000],  # 微调：5k步后启用
                                values=[0.0, 0.1],  # 5k步前权重为0，之后为0.1
                                num_frames=5,  # 窗口大小=5帧（满足1+k×4）
                                step=2,  # 步长=2，T=7产生2个窗口，支持latent_consistency
                            )
                        ),
                        # FlowLoss: 适用于视频数据（当前禁用）
                        # 微调任务：5k步后启用（微调步数较少，需提前启用）
                        # 从头训练：建议改为250k步后启用
                        # 权重说明：默认0.01较低，提高到0.05以更好捕捉AGRI数据的时序变化
                        flow=dict(
                            config=dict(
                                enabled=False,  # 禁用光流损失（RAFT模型需要3通道，AGRI数据为16通道）
                                boundaries=[5_000],  # 微调：5k步后启用
                                values=[0.0, 0.05],  # 5k步前权重为0，之后为0.05（默认0.01）
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
            # 使用gmi_loader_basic（支持NPZ格式，可以处理视频数据）
            dataset=dict(
                dataset_name="gmi_processed",  # 复用GMI数据集类（支持NPZ格式）
                # 数据路径模式（支持glob模式，包括**递归搜索）
                # 使用预处理后的视频数据：已标准化和裁剪的样本
                data_pattern="/public/share/users/sunhaofei/AGRI_processed/preprocessed_video/*_agri_video_*.npz",
                # 视频数据（每个NPZ文件包含7个时刻）
                num_video_frames=NUM_TIME_STEPS,  # 7个时刻
                use_time_sequence=False,  # 不使用时间序列（每个文件已经包含5个时刻）
                # 通道配置：根据use_mask_channel决定通道数
                target_channels=NUM_CHANNELS,  # 如果use_mask_channel=True则为10（9个AGRI通道+1个mask通道），否则为9（仅AGRI通道）
                channel_padding_mode="zero",  # 可选: "zero", "repeat", "mean"
                # 极轨卫星观测mask配置：如果启用，会在数据通道后添加一个mask通道
                # mask通道表示有效观测区域（1=有效，0=无效）
                # 注意：如果启用，target_channels应该包括mask通道（例如：14个数据通道 + 1个mask通道 = 15）
                include_mask_channel=USE_MASK_CHANNEL,  # 设置为True以启用观测mask通道，False则不使用mask通道
                # 使用预处理后的数据：跳过标准化、裁剪和缓存步骤
                use_preprocessed=True,  # 设置为True以使用预处理后的数据
                # 以下参数在use_preprocessed=True时会被忽略，但保留用于兼容性
                # 内存优化：裁剪或下采样大图像
                lat_range=None,  # AGRI数据已经裁剪到256x256，不需要纬度范围过滤
                # 确保截取到的区域至少有足够的有效观测点，否则跳过该样本
                min_valid_observations=30000,  # 最少需要30000个有效观测点（mask=1），与预处理时保持一致
                crop_size=256,  # 从中心裁剪到256x256（预处理后数据已裁剪，此参数被忽略）
                # 裁剪步长：用于匹配build_crop_cache.py生成的缓存文件
                # 必须与构建缓存时使用的stride参数一致
                stride=200,  # 与preprocess_agri_crops.py中使用的stride一致（预处理后数据不需要缓存，此参数被忽略）
            ),
            batch_size=2,  # 减少batch size以降低内存使用
            num_workers=2,  # 减少worker数量以降低内存使用（系统内存不足时）
            prefetch_factor=2,  # 减少prefetch以降低内存使用
            persistent_workers=False,  # 根据诊断结果，非持久worker性能更好
        ),
        # 验证数据加载器配置
        dataloader_val=dict(
            dataset=dict(
                dataset_name="gmi_processed",  # 复用GMI数据集类（支持NPZ格式）
                # 使用预处理后的数据：与训练保持一致
                data_pattern="/public/share/users/sunhaofei/AGRI_processed/preprocessed_video/*_agri_video_*.npz",
                num_video_frames=NUM_TIME_STEPS,  # 7个时刻
                use_time_sequence=False,  # 不使用时间序列（每个文件已经包含5个时刻）
                target_channels=NUM_CHANNELS,  # 与训练保持一致：10通道（含mask）或9通道（不含mask）
                channel_padding_mode="zero",
                include_mask_channel=USE_MASK_CHANNEL,  # 与训练保持一致
                # 使用预处理后的数据：与训练保持一致
                use_preprocessed=True,  # 设置为True以使用预处理后的数据
                # 以下参数在use_preprocessed=True时会被忽略，但保留用于兼容性
                # 与训练保持一致，使用相同的lat_range、min_valid_observations和resize
                lat_range=None,  # AGRI数据已经裁剪到256x256，不需要纬度范围过滤
                min_valid_observations=30000,  # 最少需要30000个有效观测点（mask=1），与预处理时保持一致
                crop_size=256,  # 从中心裁剪到256x256（预处理后数据已裁剪，此参数被忽略）
                # 裁剪步长：与训练保持一致
                stride=200,  # 与preprocess_agri_crops.py中使用的stride一致（预处理后数据不需要缓存，此参数被忽略）
            ),
            batch_size=2,  # 减少batch size以降低内存使用
            num_workers=2,  # 减少worker数量以降低内存使用（系统内存不足时）
            prefetch_factor=2,  # 减少prefetch以降低内存使用
            persistent_workers=False,  # 根据诊断结果，非持久worker性能更好
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
            group="agri",
            name="agri_training_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        checkpoint=dict(
            # 使用预训练的Cosmos-Tokenize1-CV4x8x8-360p checkpoint
            load_path="/public/home/sunhaofei/cosmos-predict1/checkpoints/Cosmos-Tokenize1-CV4x8x8-360p/model.pt",
            # 视频tokenizer输入形状: [B, C, T, H, W]
            # 注意：视频tokenizer需要时间维度T，C根据use_mask_channel决定（10或9），T=7，H=W=256
            jit=dict(input_shape=[1, NUM_CHANNELS, NUM_TIME_STEPS, 256, 256]),  # [B, C, T, H, W]，C=10（含mask）或9（不含mask），T=7，H=W=256
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
    name="agri_training",
    node=AGRI_TRAINING
)
