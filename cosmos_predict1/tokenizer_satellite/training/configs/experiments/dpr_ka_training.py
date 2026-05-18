# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config settings for DPR Ka tokenizer fine-tuning.

DPR Ka crop NPZ files are expected to contain:
  - dpr_ka_data: [1, 256, 256] normalized composite reflectivity
  - dpr_ka_mask: [1, 256, 256] observation mask

        The dataset emits three channels directly as:
  [composite_reflectivity, composite_reflectivity, observation_mask]
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict1.utils.lazy_config import LazyDict

DATA_CHANNELS = 2
NUM_CHANNELS = 3

DPR_KA_TRAINING: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_image"},
            {"override /data_train": "gmi_loader_basic"},
            {"override /data_val": "gmi_loader_basic"},
            {"override /loss": "video"},
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["basic"]},
            "_self_",
        ],
        model=dict(
            config=dict(
                network=dict(
                    in_channels=NUM_CHANNELS,
                    out_channels=NUM_CHANNELS,
                    spatial_compression=8,
                    patch_size=2,
                ),
                loss=dict(
                    config=dict(
                        color=dict(
                            config=dict(
                                norm="L1",
                                boundaries=[0],
                                values=[1.5],
                            )
                        ),
                        kl=dict(
                            config=dict(
                                boundaries=[0],
                                values=[1e-6],
                            )
                        ),
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
                        video_consistency=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[0.0],
                                num_frames=1,
                                step=1,
                            )
                        ),
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
                ),
            )
        ),
        dataloader_train=dict(
            dataset=dict(
                dataset_name="gmi_processed",
                data_pattern="/public/share/users/sunhaofei/yyf_data/DPR_Ka_composite_crops_2024_full_new/*_DPR_Ka.npz",
                num_video_frames=1,
                use_time_sequence=False,
                # DPR Ka already matches the pretrained tokenizer's 3-channel input.
                # Do not use ChannelAdapter; build [reflectivity, reflectivity, mask]
                # directly in GMIDataset and load the original pretrained weights.
                target_channels=None,
                channel_padding_mode="zero",
                expand_channels=DATA_CHANNELS,
                include_mask_channel=True,
                use_preprocessed=True,
                lat_range=None,
                min_valid_observations=1,
                crop_size=256,
                stride=256,
            ),
            batch_size=8,
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=False,
        ),
        dataloader_val=dict(
            dataset=dict(
                dataset_name="gmi_processed",
                data_pattern="/public/share/users/sunhaofei/yyf_data/DPR_Ka_composite_crops_2024_full_new/*_DPR_Ka.npz",
                num_video_frames=1,
                use_time_sequence=False,
                target_channels=None,
                channel_padding_mode="zero",
                expand_channels=DATA_CHANNELS,
                include_mask_channel=True,
                use_preprocessed=True,
                lat_range=None,
                min_valid_observations=1,
                crop_size=256,
                stride=256,
            ),
            batch_size=8,
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=False,
        ),
        optimizer=dict(
            lr=5e-5,
            betas=(0.5, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        ),
        scheduler=dict(
            warmup=10000,
        ),
        job=dict(
            project="satellite_tokenizer",
            group="dpr_ka",
            name="dpr_ka_tokenizer_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        checkpoint=dict(
            load_path="/public/home/sunhaofei/cosmos-predict1/checkpoints/Cosmos-Tokenize1-CI8x8-360p/model.pt",
            jit=dict(input_shape=[1, NUM_CHANNELS, 256, 256]),
        ),
    )
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="dpr_ka_training",
    node=DPR_KA_TRAINING,
)
