"""Experiment config for CMA-MESO CV (Continuous Video) Tokenizer fine-tuning.

CMA-MESO is a regional numerical weather prediction model, with multi-variable,
potentially multi-level meteorological fields. This config fine-tunes the
Cosmos CV4x8x8-360p tokenizer to encode MESO data into a continuous latent space.

Architecture: CausalContinuousVideoTokenizer (CV4x8x8)
  - Temporal compression: 4x  (e.g., T=9 -> latent T=3)
  - Spatial compression:  8x  (e.g., 256x256 -> 32x32)
  - Latent channels:      16

Input shape: [B, C, T, H, W]
  - C = NUM_CHANNELS (number of meteorological variables/levels)
  - T = NUM_TIME_STEPS (must satisfy 1 + 4k for temporal compression, e.g., 5, 9, 13, 17)
  - H, W = 256 (spatial crop size)

Latent shape: [B, 16, ceil(T/4), H/8, W/8]
  - e.g., [B, 16, 3, 32, 32] for T=9, H=W=256

Example usage:
    # From MESO-Tokenizer directory:
    cd /public/home/sunhaofei/cosmos-predict1
    PYTHONPATH=$(pwd):/public/home/sunhaofei/MESO-Tokenizer \
    torchrun --nproc_per_node=4 \
        -m cosmos_predict1.tokenizer_satellite.training.train \
        --config=cosmos_predict1/tokenizer_satellite/training/configs/config.py \
        -- experiment=meso_cv_training \
        dataloader_train.dataset.data_pattern="/path/to/MESO_preprocessed/train/*.npz"
"""

from hydra.core.config_store import ConfigStore
from cosmos_predict1.utils.lazy_config import LazyDict

# ============================================================================
# CMA-MESO data configuration
# ============================================================================
# Number of meteorological channels (variables x levels).
# Adjust this based on your actual data:
#   - 4 channels:  surface only (prmsl, 2t, 10u, 10v)
#   - ~20 channels: surface + 4 pressure levels x 4 variables
#   - ~40 channels: surface + multi-level, multi-variable
NUM_CHANNELS = 4

# Number of time steps per training sample.
# Must satisfy T = 1 + 4k for CV4x8x8 temporal compression.
#   T=5  -> latent_T=2
#   T=9  -> latent_T=3
#   T=13 -> latent_T=4
#   T=17 -> latent_T=5
NUM_TIME_STEPS = 9

# Spatial crop size (H = W = CROP_SIZE)
CROP_SIZE = 256

# ============================================================================
# Training configuration
# ============================================================================
MESO_CV_TRAINING: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_factorized_video"},
            {"override /loss": "video"},
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["basic"]},
            {"override /data_train": "meso_loader_basic"},
            {"override /data_val": "meso_loader_basic"},
            "_self_",
        ],
        # ---- Model (network) config ----
        model=dict(
            config=dict(
                network=dict(
                    in_channels=NUM_CHANNELS,
                    out_channels=NUM_CHANNELS,
                    # CV4x8x8 architecture settings
                    temporal_compression=4,
                    spatial_compression=8,
                    patch_size=2,
                ),
                loss=dict(
                    config=dict(
                        # L1 reconstruction loss - primary loss for weather data
                        color=dict(
                            config=dict(
                                norm="L1",
                                boundaries=[0],
                                values=[1.5],  # Stronger reconstruction emphasis
                            )
                        ),
                        # KL regularization (very small weight, keeps latent space regular)
                        kl=dict(
                            config=dict(
                                boundaries=[0],
                                values=[1e-6],
                            )
                        ),
                        # Perceptual loss: DISABLED for non-RGB data
                        # VGG-based LPIPS is meaningless for meteorological fields
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
                        # Video consistency loss: enforce temporal coherence
                        # Important for weather field encoding - physical fields are
                        # temporally smooth, consistency loss helps capture this.
                        video_consistency=dict(
                            config=dict(
                                enabled=True,
                                boundaries=[5_000],
                                values=[0.0, 0.1],  # Enable after 5k warmup steps
                                num_frames=5,  # Window size for consistency
                                step=2,  # Stride between windows
                            )
                        ),
                        # Flow loss: DISABLED (requires 3-channel RGB input)
                        flow=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[5_000],
                                values=[0.0, 0.05],
                                scale=2,
                                dtype="bfloat16",
                                checkpoint_activations=False,
                            )
                        ),
                    )
                ),
            )
        ),
        # ---- Data loaders ----
        dataloader_train=dict(
            dataset=dict(
                dataset_name="meso",
                data_pattern="/path/to/MESO_preprocessed/train/*.npz",  # Override in run script
                num_video_frames=NUM_TIME_STEPS,
                use_time_sequence=False,  # Each NPZ already contains full video
                target_channels=NUM_CHANNELS,
                channel_padding_mode="zero",
                include_mask_channel=False,  # MESO data has no observation gaps
                use_preprocessed=True,
                lat_range=None,
                min_valid_observations=0,
                crop_size=CROP_SIZE,
                stride=128,
            ),
            batch_size=2,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=False,
        ),
        dataloader_val=dict(
            dataset=dict(
                dataset_name="meso",
                data_pattern="/path/to/MESO_preprocessed/val/*.npz",  # Override in run script
                num_video_frames=NUM_TIME_STEPS,
                use_time_sequence=False,
                target_channels=NUM_CHANNELS,
                channel_padding_mode="zero",
                include_mask_channel=False,
                use_preprocessed=True,
                lat_range=None,
                min_valid_observations=0,
                crop_size=CROP_SIZE,
                stride=128,
            ),
            batch_size=2,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=False,
        ),
        # ---- Optimizer: fine-tuning LR (half of pretraining) ----
        optimizer=dict(
            lr=5e-5,
            betas=(0.5, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        ),
        # ---- LR scheduler: longer warmup for stable fine-tuning ----
        scheduler=dict(
            warmup=10000,
        ),
        # ---- Job metadata ----
        job=dict(
            project="meso_tokenizer",
            group="cv",
            name="meso_cv_training_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        # ---- Checkpoint: load pretrained CV4x8x8-360p ----
        checkpoint=dict(
            load_path="/public/home/sunhaofei/cosmos-predict1/checkpoints/Cosmos-Tokenize1-CV4x8x8-360p/model.pt",
            jit=dict(
                input_shape=[1, NUM_CHANNELS, NUM_TIME_STEPS, CROP_SIZE, CROP_SIZE],
            ),
            # Channel init: cycle pretrained 3-channel weights to fill NUM_CHANNELS
            channel_init_strategy="all_pretrained",
        ),
    )
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="meso_cv_training",
    node=MESO_CV_TRAINING,
)
