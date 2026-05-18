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

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict1.diffusion.training.callbacks.iter_speed import IterSpeed
from cosmos_predict1.diffusion.training.callbacks.low_precision import LowPrecisionCallback
from cosmos_predict1.diffusion.training.datasets.dataset_video import Dataset
from cosmos_predict1.diffusion.training.models.extend_model import FSDPExtendDiffusionModel
from cosmos_predict1.diffusion.training.models.model_peft import PEFTExtendDiffusionModel
from cosmos_predict1.diffusion.training.networks.general_dit_lvg import VideoExtendGeneralDIT
from cosmos_predict1.diffusion.training.utils.peft.lora_config import get_fa_ca_qv_lora_config
from cosmos_predict1.utils import log
from cosmos_predict1.utils.callback import ProgressBarCallback
from cosmos_predict1.utils.callbacks.grad_clip import GradClip
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict

# 创建一个分布式采样器，用于多 GPU 训练（DistributedSampler），
# 确保数据在不同进程间均匀分布，支持 shuffling 和固定种子（seed=0）。
def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )

# 初始化 Hydra 配置存储，用于注册实验配置
cs = ConfigStore.instance()

n_length = 15
num_frames = 8 * n_length + 1  # 121  # 每个样本包含 121 帧视频序列（时间维度较长，适合长视频预测）。

# HDVILA example， HDVILA 数据集，高清视频数据集
# 分辨率 (720, 1280)，121 帧，sequence_interval=1（连续帧）。
example_video_dataset_hdvila = L(Dataset)(
    dataset_dir="datasets/hdvila",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1280),
    start_frame_interval=1,
)

dataloader_train_hdvila = L(DataLoader)(
    dataset=example_video_dataset_hdvila,
    sampler=L(get_sampler)(dataset=example_video_dataset_hdvila),
    batch_size=1,     # 小批量，适合大模型训练以节省内存）。
    drop_last=True,   # 丢弃不完整的批次。
    pin_memory=True,  # 加速 CPU 到 GPU 数据传输。
    num_workers=8,    # 多线程数据加载器，使用 8 个工作线程。
)

# Cosmos-NeMo-Assets example
example_video_dataset_cosmos_nemo_assets = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1280),
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

# 低分辨率变体（如 _4gpu_80gb, _8gpu_40gb, _4gpu_40gb）：降低分辨率（如 (384, 384) 或 (192, 192)）和帧数（如 121 → 25）
# Cosmos-NeMo-Assets examples with more affordable GPUs setup (4 GPUs or 40GB VRAM)
n_length_4gpu_80gb = 15
num_frames_4gpu_80gb = 8 * n_length_4gpu_80gb + 1  # 121
example_video_dataset_cosmos_nemo_assets_4gpu_80gb = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames_4gpu_80gb,
    video_size=(384, 384),  # a low-res example for lower VRAM utilization without considering the content aspect ratio.
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_4gpu_80gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_4gpu_80gb,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_4gpu_80gb),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

n_length_8gpu_40gb = 3
num_frames_8gpu_40gb = 8 * n_length_8gpu_40gb + 1  # 25
example_video_dataset_cosmos_nemo_assets_8gpu_40gb = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames_8gpu_40gb,
    video_size=(384, 384),  # a low-res example for lower VRAM utilization without considering aspect ratio.
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_8gpu_40gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_8gpu_40gb,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_8gpu_40gb),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

n_length_4gpu_40gb = 3
num_frames_4gpu_40gb = 8 * n_length_4gpu_40gb + 1  # 25
example_video_dataset_cosmos_nemo_assets_4gpu_40gb = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames_4gpu_40gb,
    video_size=(192, 192),  # a low-res example for lower VRAM utilization without considering aspect ratio.
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_4gpu_40gb = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_4gpu_40gb,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_4gpu_40gb),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)

#_480_848 变体：分辨率 (480, 848)，用于 LoRA 示例。
# Cosmos-NeMo-Assets 480x848 example for lora
example_video_dataset_cosmos_nemo_assets_480_848 = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(480, 848),
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets_480_848 = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_480_848,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_480_848),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

dataloader_val_cosmos_nemo_assets_480_848 = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_480_848,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_480_848),
    batch_size=1,
    drop_last=True,
    pin_memory=True,
    num_workers=8,
)

# 代码注册了 6 个实验变体，使用 LazyDict（延迟加载配置），
# 每个变体是一个字典，定义了模型、优化器、检查点、训练器等。它们共享许多配置，但针对不同数据集和硬件优化。
video2world_7b_example_hdvila = LazyDict(
    dict(
        # 覆盖默认组件
        defaults=[
            {"override /net": "faditv2_7b"},          # 7B 参数模型
            {"override /conditioner": "video_cond"},  # 用视频作为条件信息
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},  # 8x8x8 压缩的扩散 tokenizer
            "_self_",
        ],
        # 实验元数据
        job=dict(
            project="posttraining",                # 项目名 ("posttraining")
            group="diffusion_video2world",         # 组名 ("diffusion_video2world")
            name="video2world_7b_example_hdvila",  # 名称 ("video2world_7b_example_hdvila")
        ),
        
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,                            # 保存间隔
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,                     # 允许部分加载
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=2000,                           # 最大迭代次数
            distributed_parallelism="fsdp",          # 分布式并行方式 (FSDP)
            logging_iter=200,                        # 日志记录间隔
            callbacks=dict(                          
                grad_clip=L(GradClip)(model_key="model",fsdp_enabled=True,),    # 梯度裁剪回调
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1), # 低精度回调
                iter_speed=L(IterSpeed)(every_n=10, hit_thres=0,),  # 迭代速度监控
                progress_bar=L(ProgressBarCallback)(),              # 进度条
            ),
        ),
        # 模型并行设置
        model_parallel=dict(  
            sequence_parallel=False,              # 序列并行（不启用）
            tensor_model_parallel_size=1,         # 张量模型并行大小
            context_parallel_size=1,              # 上下文模型并行大小
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                88,  # Latent height dim
                160,  # Latent width dim
            ],
            loss_reduce="mean",         
            ema=dict(enabled=True,),      # 指数移动平均，False 时节省内存。
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(           # 核心网络，扩散 Transformer (DiT)，支持 RoPE 位置编码扩展
                extra_per_block_abs_pos_emb=True,
                extra_per_block_abs_pos_emb_type="learnable",
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",                               # 条件位置
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,                              # 最多 2 帧条件
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],                # 预热步数
            cycle_lengths=[10000000000000],      # 循环长度（非常大，表示无限循环）
            f_start=[1.0e-6],                    # 初始学习率
            f_max=[1.0],                         # 最大学习率
            f_min=[1.0],                         # 最小学习率
        ),
        dataloader_train=dataloader_train_hdvila,
    )
)


video2world_7b_example_cosmos_nemo_assets = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=2000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                88,  # Latent height dim
                160,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets,
    )
)

video2world_7b_example_cosmos_nemo_assets_4gpu_80gb = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_4gpu_80gb",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
        ),
        trainer=dict(
            max_iter=2000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=True,
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_4gpu_80gb,
                spatial_resolution="384",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_4gpu_80gb,
    )
)

video2world_7b_example_cosmos_nemo_assets_8gpu_40gb = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_8gpu_40gb",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=2000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                48,  # Latent height dim
                48,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_8gpu_40gb,
                spatial_resolution="384",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_8gpu_40gb,
    )
)

video2world_7b_example_cosmos_nemo_assets_4gpu_40gb = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_example_cosmos_nemo_assets_4gpu_40gb",
        ),
        optimizer=dict(
            lr=2 ** (-14.3),  # 2**(-14.3) approx 5e-5
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=200,
            broadcast_via_filesystem=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=2000,
            distributed_parallelism="fsdp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=True,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        ),
        model=dict(
            latent_shape=[
                16,  # Latent channel dim
                16,  # Latent temporal dim
                24,  # Latent height dim
                24,  # Latent width dim
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=True,
            fsdp=dict(
                policy="block",
                checkpoint=True,
                min_num_params=1024,
                sharding_group_size=32,
                sharding_strategy="hybrid",
            ),
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(
                pixel_chunk_duration=num_frames_4gpu_40gb,
                spatial_resolution="192",
            ),
        ),
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        # warming up for first 2500 steps
        scheduler=dict(
            warm_up_steps=[2500],
            cycle_lengths=[10000000000000],
            f_start=[1.0e-6],
            f_max=[1.0],
            f_min=[1.0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_4gpu_40gb,
    )
)

# LoRA 变体，使用 PEFT (peft_control=get_fa_ca_qv_lora_config，rank=8)，
# distributed_parallelism="ddp"，context_parallel_size=4，lr=1e-4，无 warmup，max_iter=5000，
# 有验证 DataLoader。
video2world_7b_lora_example_cosmos_nemo_assets = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /conditioner": "video_cond"},
            {"override /ckpt_klass": "peft"},
            {"override /checkpoint": "local"},
            {"override /vae": "cosmos_diffusion_tokenizer_comp8x8x8"},
            "_self_",
        ],
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_7b_lora_example_cosmos_nemo_assets",
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=0.1,
            betas=[0.9, 0.99],
            eps=1e-10,
        ),
        checkpoint=dict(
            save_iter=1000,
            broadcast_via_filesystem=True,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt",
            load_training_state=False,
            strict_resume=False,
            keys_not_to_resume=[],
            async_saving=False,  # set to False to save memory
        ),
        trainer=dict(
            max_iter=5000,
            distributed_parallelism="ddp",
            logging_iter=200,
            callbacks=dict(
                grad_clip=L(GradClip)(
                    model_key="model",
                    fsdp_enabled=False,
                ),
                low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
                iter_speed=L(IterSpeed)(
                    every_n=10,
                    hit_thres=0,
                ),
                progress_bar=L(ProgressBarCallback)(),
            ),
        ),
        model_parallel=dict(
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=4,
        ),
        model=dict(
            peft_control=get_fa_ca_qv_lora_config(first_nblocks=28, rank=8, scale=1),
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
            loss_reduce="mean",
            ema=dict(
                enabled=False,  # turn off to save memory
            ),
            fsdp_enabled=False,
            net=L(VideoExtendGeneralDIT)(
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=2,
            ),
            adjust_video_noise=True,
            conditioner=dict(
                video_cond_bool=dict(
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma",
                    condition_on_augment_sigma=False,
                    dropout_rate=0.0,  # No dropout
                    first_random_n_num_condition_t_max=2,
                    normalize_condition_latent=False,
                    # Let the augment sigma mostly fall in the range of 0 to 0.3
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                )
            ),
            vae=dict(pixel_chunk_duration=num_frames),
        ),
        model_obj=L(PEFTExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        scheduler=dict(
            warm_up_steps=[0],
        ),
        dataloader_train=dataloader_train_cosmos_nemo_assets_480_848,
        dataloader_val=dataloader_val_cosmos_nemo_assets_480_848,
    )
)

# 将所有变体注册到 Hydra 配置组 "experiment"，允许通过命令行运行特定实验（如 hydra experiment=video2world_7b_example_hdvila）。
def register_experiments(cs):
    # Register the experiments
    for _item in [
        video2world_7b_example_hdvila,
        video2world_7b_example_cosmos_nemo_assets,
        video2world_7b_example_cosmos_nemo_assets_4gpu_80gb,
        video2world_7b_example_cosmos_nemo_assets_8gpu_40gb,
        video2world_7b_example_cosmos_nemo_assets_4gpu_40gb,
        video2world_7b_lora_example_cosmos_nemo_assets,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )


# 总体，这段代码是为 Cosmos-Predict1-7B-Video2World 模型的微调/后训练（post-training）设计的，支持单视角视频预测，强调硬件灵活性和高效训练（FSDP, LoRA）。