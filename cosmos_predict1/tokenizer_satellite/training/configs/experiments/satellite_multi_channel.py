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

"""Config settings for multi-channel satellite tokenizer training (方案A: 最大通道数统一训练)

This configuration implements Solution A: Unified training with maximum channel count.
All satellite instruments are trained together with their channels padded/repeated
to match the maximum channel number across all instruments.

Example usage:
    python train.py experiment=satellite_multi_channel \
        dataloader_train.dataset.target_channels=18 \
        model.config.network.in_channels=18 \
        model.config.network.out_channels=18
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict1.utils.lazy_config import LazyDict

# 方案A: 最大通道数统一训练配置
# 假设有多个卫星仪器，通道数分别为: instrument1=13, instrument2=18, instrument3=10
# 最大通道数为18，所有数据统一填充到18通道进行训练
SATELLITE_MULTI_CHANNEL: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_factorized_video"},
            {"override /data_train": "hdvila_video720"},
            {"override /data_val": "hdvila_video720"},
            {"override /loss": "video"},
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["basic"]},
            "_self_",
        ],
        # 模型配置：设置输入输出通道数为最大通道数
        model=dict(
            config=dict(
                network=dict(
                    # 关键配置：设置输入输出通道数为最大通道数（例如18）
                    # 如果启用include_mask_channel=True，需要+1（例如：18数据通道 + 1mask通道 = 19）
                    in_channels=18,  # 根据实际最大通道数调整（如果启用mask，需要+1）
                    out_channels=18,  # 根据实际最大通道数调整（如果启用mask，需要+1）
                ),
                loss=dict(
                    config=dict(
                        perceptual=dict(
                            config=dict(
                                lpips_boundaries=[0],
                                lpips_values=[0.1],
                                gram_enabled=False,
                                gram_boundaries=[0],
                            )
                        ),
                        video_consistency=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[1.0],
                                num_frames=32,
                                step=8,
                            )
                        ),
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
            dataset=dict(
                crop_height=256,
                num_video_frames=49,
                # 关键配置：设置目标通道数为最大通道数
                target_channels=18,  # 根据实际最大通道数调整
                channel_padding_mode="zero",  # 可选: "zero", "repeat", "mean"
                # 极轨卫星观测mask配置：如果启用，会在数据通道后添加一个mask通道
                # mask通道表示有效观测区域（1=有效，0=无效）
                # 注意：如果启用，target_channels应该包括mask通道（例如：18个数据通道 + 1个mask通道 = 19）
                include_mask_channel=False,  # 设置为True以启用观测mask通道
            ),
            batch_size=1,
        ),
        # 验证数据加载器配置
        dataloader_val=dict(
            dataset=dict(
                crop_height=720,
                num_video_frames=49,
                # 验证时也使用相同的目标通道数
                target_channels=18,  # 根据实际最大通道数调整（如果启用mask，需要+1）
                channel_padding_mode="zero",
                include_mask_channel=False,  # 与训练保持一致
            ),
            batch_size=1,
        ),
        job=dict(
            project="satellite_tokenizer",
            group="multi_channel",
            name="unified_max_channels_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        checkpoint=dict(
            load_path=None,  # 可以设置为预训练checkpoint路径，模型会自动适配通道数
            jit=dict(input_shape=[1, 18, 17, 512, 512]),  # [B, C, T, H, W]
        ),
    )
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="satellite_multi_channel",
    node=SATELLITE_MULTI_CHANNEL
)

