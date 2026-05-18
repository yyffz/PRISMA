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

"""dataloader config options

Available dataloader options:
    image_loader_basic
    video_loader_basic
    joint_image_video_loader_basic
"""

from torch.utils.data import DataLoader

from cosmos_predict1.tokenizer_satellite.training.configs.base.mock_data import get_mock_video_dataloader
from cosmos_predict1.tokenizer_satellite.training.datasets.dataset_provider import dataset_entry
from cosmos_predict1.utils.lazy_config import LazyCall

DATALOADER_OPTIONS = {}


def dataloader_register(key):
    def decorator(func):
        DATALOADER_OPTIONS[key] = func
        return func

    return decorator


@dataloader_register("video_loader_basic")
def get_video_dataloader(
    dataset_name,
    is_train,
    batch_size=1,
    num_video_frames=25,
    resolution="720",
    crop_height=128,
    num_workers=8,
    target_channels=None,
    channel_padding_mode="zero",
    include_mask_channel=False,
):
    """Get video dataloader with optional channel adaptation.
    
    Args:
        dataset_name: Name of the dataset
        is_train: Whether this is training data
        batch_size: Batch size
        num_video_frames: Number of video frames
        resolution: Video resolution
        crop_height: Crop height
        num_workers: Number of data loading workers
        target_channels: Target number of channels for unified training.
            If None, no channel adaptation is applied.
            If include_mask_channel=True, this should include the mask channel.
        channel_padding_mode: How to pad channels ("zero", "repeat", "mean")
        include_mask_channel: If True, adds an observation mask channel as the last channel.
            Useful for polar-orbiting satellites where some regions have no valid observations.
    """
    if dataset_name.startswith("mock"):
        return get_mock_video_dataloader(
            batch_size=batch_size,
            is_train=is_train,
            num_video_frames=num_video_frames,
            resolution=resolution,
            crop_height=crop_height,
        )
    return LazyCall(DataLoader)(
        dataset=LazyCall(dataset_entry)(
            dataset_name=dataset_name,
            dataset_type="video",
            is_train=is_train,
            resolution=resolution,
            crop_height=crop_height,
            num_video_frames=num_video_frames,
            target_channels=target_channels,
            channel_padding_mode=channel_padding_mode,
            include_mask_channel=include_mask_channel,
        ),
        batch_size=batch_size,  # 2
        num_workers=num_workers,  # 8
        prefetch_factor=2,
        shuffle=None,  # do we need this?
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
    )


@dataloader_register("gmi_loader_basic")
def get_gmi_dataloader(
    dataset_name="gmi_processed",
    is_train=True,
    batch_size=1,
    num_video_frames=1,
    resolution="720",
    crop_height=128,
    num_workers=8,
    target_channels=None,
    channel_padding_mode="zero",
    include_mask_channel=False,
    use_time_sequence=False,
    data_pattern=None,
    resize=None,
    crop_size=None,
    lat_range=None,
    min_valid_observations=1000,
    stride=1,
    use_preprocessed=False,
    expand_channels=None,
):
    """Create a DataLoader for GMI processed NPZ files.
    
    Args:
        dataset_name: Name of the dataset (used for pattern lookup if data_pattern is None)
        is_train: Whether this is training data
        batch_size: Batch size
        num_video_frames: Number of video frames (only used if use_time_sequence=True)
        resolution: Video resolution (for augmentation config, not used for GMI)
        crop_height: Crop height (for augmentation config, not used for GMI)
        num_workers: Number of data loading workers
        target_channels: Target number of channels for unified training.
            If None, no channel adaptation is applied.
            If include_mask_channel=True, this should include the mask channel.
        channel_padding_mode: How to pad channels ("zero", "repeat", "mean")
        include_mask_channel: If True, adds an observation mask channel as the last channel.
            Useful for polar-orbiting satellites where some regions have no valid observations.
        use_time_sequence: If True, load multiple consecutive files to form a time sequence.
        data_pattern: Custom data pattern (overrides dataset_name lookup).
            Example: "datasets/GMI_processed/**/*_GMI.npz"
        lat_range: Latitude range to keep, e.g., (-65, 65) for 65°N-65°S.
            If None, no latitude cropping is applied.
        min_valid_observations: Minimum number of valid observations required in the cropped region.
            If the number of valid observations is less than this threshold, the sample will be skipped.
            Default is 1000.
        stride: Stride used when building the crop regions cache. This should match the stride
            used in build_crop_cache.py to ensure cache compatibility. Default is 1.
        use_preprocessed: Whether NPZ files already contain cropped/normalized arrays.
        expand_channels: If set, repeat data channels up to this number before mask handling.
    """
    return LazyCall(DataLoader)(
        dataset=LazyCall(dataset_entry)(
            dataset_name=dataset_name,
            dataset_type="gmi",
            is_train=is_train,
            resolution=resolution,
            crop_height=crop_height,
            num_video_frames=num_video_frames,
            target_channels=target_channels,
            channel_padding_mode=channel_padding_mode,
            include_mask_channel=include_mask_channel,
            use_time_sequence=use_time_sequence,
            data_pattern=data_pattern,
            resize=resize,
            crop_size=crop_size,
            lat_range=lat_range,
            min_valid_observations=min_valid_observations,
            stride=stride,
            use_preprocessed=use_preprocessed,
            expand_channels=expand_channels,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
        shuffle=None,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
    )


@dataloader_register("era5_loader_basic")
def get_era5_dataloader(
    dataset_name="era5",
    is_train=True,
    batch_size=4,
    num_video_frames=1,
    resolution="720",
    crop_height=136,
    num_workers=4,
    target_channels=15,
    channel_padding_mode="zero",
    include_mask_channel=False,
    data_pattern=None,
    crop_size=None,
    **kwargs,
):
    """Create a DataLoader for ERA5 preprocessed per-frame NPZ files."""
    return LazyCall(DataLoader)(
        dataset=LazyCall(dataset_entry)(
            dataset_name=dataset_name,
            dataset_type="era5",
            is_train=is_train,
            resolution=resolution,
            crop_height=crop_height,
            num_video_frames=num_video_frames,
            target_channels=target_channels,
            channel_padding_mode=channel_padding_mode,
            include_mask_channel=include_mask_channel,
            data_pattern=data_pattern,
            crop_size=crop_size,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
        shuffle=None,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
    )


@dataloader_register("meso_loader_basic")
def get_meso_dataloader(
    dataset_name="meso",
    is_train=True,
    batch_size=2,
    num_video_frames=9,
    resolution="720",
    crop_height=256,
    num_workers=4,
    target_channels=4,
    channel_padding_mode="zero",
    include_mask_channel=False,
    data_pattern=None,
    crop_size=256,
    **kwargs,
):
    """Create a DataLoader for CMA-MESO preprocessed NPZ files."""
    return LazyCall(DataLoader)(
        dataset=LazyCall(dataset_entry)(
            dataset_name=dataset_name,
            dataset_type="meso",
            is_train=is_train,
            resolution=resolution,
            crop_height=crop_height,
            num_video_frames=num_video_frames,
            target_channels=target_channels,
            channel_padding_mode=channel_padding_mode,
            include_mask_channel=include_mask_channel,
            data_pattern=data_pattern,
            crop_size=crop_size,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
        shuffle=None,
        sampler=None,
        persistent_workers=False,
        pin_memory=True,
    )
