"""Experiment config for ERA5 CI8x8 (Continuous Image) Tokenizer fine-tuning.

Fine-tunes Cosmos-Tokenize1-CI8x8-360p on ERA5 reanalysis data (15 channels).
CI8x8 encodes each frame independently (no temporal compression), which is
ideal for ERA5's slowly-varying meteorological fields at 1-hour intervals.

Architecture: ContinuousImageTokenizer (CI8x8)
  - Spatial compression: 8x  (136×200 → 17×25)
  - No temporal compression (each hour encoded independently)
  - Latent channels: 16

Input shape:  [B, C=15, H=136, W=200]   (single frame)
Latent shape: [B, 16, 17, 25]           (per frame)

After stacking 13 hours: [B, 16, 13, 17, 25]  (done at encoding time, not training)

Example usage:
    cd /public/home/sunhaofei/cosmos-predict1
    PYTHONPATH=$(pwd):/public/home/sunhaofei/ERA5-Tokenizer \
    torchrun --nproc_per_node=4 \
        -m cosmos_predict1.tokenizer_satellite.training.train \
        --config=cosmos_predict1/tokenizer_satellite/training/configs/config.py \
        -- experiment=era5_ci_training \
        dataloader_train.dataset.data_pattern="/path/to/ERA5_preprocessed/train/*.npz"
"""

from hydra.core.config_store import ConfigStore
from cosmos_predict1.utils.lazy_config import LazyDict

NUM_CHANNELS = 15   # 4 surface + 11 atmospheric (selected levels)
CROP_H = 136        # lat: 15°-50°N, cropped to 8x multiple
CROP_W = 200        # lon: 85°-135°E, cropped to 8x multiple

ERA5_CI_TRAINING: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_image"},
            {"override /loss": "video"},
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["basic"]},
            {"override /data_train": "era5_loader_basic"},
            {"override /data_val": "era5_loader_basic"},
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
                        # LPIPS disabled: VGG not meaningful for meteorological fields
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
                        # Video consistency: disabled for CI (single-frame)
                        video_consistency=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[0.0],
                            )
                        ),
                        # Flow loss: disabled
                        flow=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[0.0],
                            )
                        ),
                    )
                ),
            )
        ),
        dataloader_train=dict(
            dataset=dict(
                dataset_name="era5",
                data_pattern="/path/to/ERA5_preprocessed/train/*.npz",
                num_video_frames=1,
                use_time_sequence=False,
                target_channels=NUM_CHANNELS,
                channel_padding_mode="zero",
                include_mask_channel=False,
                use_preprocessed=True,
                lat_range=None,
                min_valid_observations=0,
                crop_size=None,  # No random crop, use full 136×200
                stride=1,
            ),
            batch_size=4,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=False,
        ),
        dataloader_val=dict(
            dataset=dict(
                dataset_name="era5",
                data_pattern="/path/to/ERA5_preprocessed/val/*.npz",
                num_video_frames=1,
                use_time_sequence=False,
                target_channels=NUM_CHANNELS,
                channel_padding_mode="zero",
                include_mask_channel=False,
                use_preprocessed=True,
                lat_range=None,
                min_valid_observations=0,
                crop_size=None,
                stride=1,
            ),
            batch_size=4,
            num_workers=4,
            prefetch_factor=2,
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
            project="era5_tokenizer",
            group="ci",
            name="era5_ci_training_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        checkpoint=dict(
            load_path="/public/home/sunhaofei/cosmos-predict1/checkpoints/Cosmos-Tokenize1-CI8x8-360p/model.pt",
            jit=dict(
                input_shape=[1, NUM_CHANNELS, CROP_H, CROP_W],
            ),
            channel_init_strategy="all_pretrained",
        ),
    )
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="era5_ci_training",
    node=ERA5_CI_TRAINING,
)
