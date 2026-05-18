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

"""Implementations of dataset settings and augmentations for tokenization

Run this command to interactively debug:
PYTHONPATH=$(pwd) python -m \
cosmos_predict1.tokenizer.training.datasets.dataset_provider \
    --dataset_name hdvila_video \
    --is_train
"""

from cosmos_predict1.tokenizer_satellite.training.datasets.augmentation_provider import (
    video_train_augmentations,
    video_val_augmentations,
)
from cosmos_predict1.tokenizer_satellite.training.datasets.channel_adapter import ChannelAdapter
from cosmos_predict1.tokenizer_satellite.training.datasets.gmi_dataset import GMIDataset
from cosmos_predict1.tokenizer_satellite.training.datasets.utils import categorize_aspect_and_store
from cosmos_predict1.tokenizer_satellite.training.datasets.video_dataset import Dataset
from cosmos_predict1.utils.lazy_config import instantiate

_VIDEO_PATTERN_DICT = {
    "hdvila_video": "datasets/hdvila/videos/*.mp4",
}

_GMI_PATTERN_DICT = {
    "gmi_processed": "datasets/GMI_processed/**/*_GMI.npz",
}


def apply_augmentations(data_dict, augmentations_dict):
    """
    Loop over each LazyCall object and apply it to data_dict in place.
    """
    for aug_name, lazy_aug in augmentations_dict.items():
        aug_instance = instantiate(lazy_aug)
        data_dict = aug_instance(data_dict)
    return data_dict


class AugmentDataset(Dataset):
    def __init__(self, base_dataset, augmentations_dict, channel_adapter=None):
        """
        base_dataset: the video dataset instance
        augmentations_dict: the dictionary returned by
                            video_train_augmentations() or video_val_augmentations()
        channel_adapter: Optional ChannelAdapter instance for multi-channel adaptation
        """
        self.base_dataset = base_dataset
        self.channel_adapter = channel_adapter

        # Pre-instantiate every augmentation ONCE:
        self.augmentations = []
        for aug_name, lazy_aug in augmentations_dict.items():
            aug_instance = instantiate(lazy_aug)  # build the actual augmentation
            self.augmentations.append((aug_name, aug_instance))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        # Get the raw sample from the base dataset
        data = self.base_dataset[index]
        # print(f'dataset_providerd data["video"].shape: {data["video"].shape}')
        
        # Store original channel number for potential recovery
        original_channels = data["video"].shape[0]
        data["original_channels"] = original_channels
        
        # Get observation mask if available (for polar-orbiting satellite data)
        # The mask should indicate valid observation regions (1=valid, 0=invalid)
        observation_mask = data.get("observation_mask", None)
        
        # Apply channel adaptation if adapter is provided
        if self.channel_adapter is not None:
            data["video"] = self.channel_adapter.adapt(data["video"], observation_mask=observation_mask)
            # print(f'data["video"].shape after channel adaptation: {data["video"].shape}')
        
        data = categorize_aspect_and_store(data)
        # print(f'dataset_providerd data["video"].shape after categorize_aspect_and_store: {data["video"].shape}')
        # # Apply each pre-instantiated augmentation
        # for aug_name, aug_instance in self.augmentations:
        #     data = aug_instance(data)
        #     print(f'data["video"].shape after {aug_name}: {data["video"].shape}')
        return data


def dataset_entry(
    dataset_name: str,
    dataset_type: str,
    is_train: bool = True,
    resolution="720",
    crop_height=256,
    num_video_frames=25,
    target_channels: int = None,
    channel_padding_mode: str = "zero",
    include_mask_channel: bool = False,
    use_time_sequence: bool = False,
    data_pattern: str = None,
    resize: int = None,
    crop_size: int = None,
    lat_range: tuple = None,
    min_valid_observations: int = 1000,
    stride: int = 1,
    use_preprocessed: bool = False,
    expand_channels: int = None,
) -> AugmentDataset:
    """Create dataset with optional channel adaptation.
    
    Args:
        dataset_name: Name of the dataset
        dataset_type: Type of dataset (e.g., "video", "gmi")
        is_train: Whether this is training data
        resolution: Video resolution
        crop_height: Crop height
        num_video_frames: Number of video frames
        target_channels: Target number of channels for unified training.
            If None, no channel adaptation is applied.
            If include_mask_channel=True, this should include the mask channel.
        channel_padding_mode: How to pad channels ("zero", "repeat", "mean")
        include_mask_channel: If True, adds an observation mask channel as the last channel.
            Useful for polar-orbiting satellites where some regions have no valid observations.
            The mask channel indicates valid observation regions (1 for valid, 0 for invalid).
        use_time_sequence: For GMI dataset, whether to load multiple consecutive files as time sequence.
        data_pattern: Custom data pattern (overrides dataset_name lookup).
        use_preprocessed: For GMI dataset, whether to use preprocessed data (already normalized and cropped).
            If True, NPZ files should contain gmi_data [13, H, W] and observation_mask [H, W].
            All preprocessing steps (normalization, cropping, lat_range filtering) will be skipped.
        expand_channels: If specified, expands single-channel data to the specified number of channels.
            Mainly used for IMERG data (1 channel -> 3 channels).
    
    Returns:
        AugmentDataset with optional channel adaptation
    """
    # Instantiate the base dataset based on dataset_type
    if dataset_type.lower() == "gmi":
        # GMI NPZ dataset
        if data_pattern is None:
            data_pattern = _GMI_PATTERN_DICT.get(dataset_name.lower(), "datasets/GMI_processed/**/*_GMI.npz")

        # 判断是否为图像模式（通过dataset_type判断）
        is_image_mode = (dataset_type.lower() == "gmi" and not use_time_sequence)
        
        base_dataset = GMIDataset(
            data_pattern=data_pattern,
            num_video_frames=num_video_frames if use_time_sequence else 1,
            use_time_sequence=use_time_sequence,
            is_image_mode=is_image_mode,
            resize=resize,
            crop_size=crop_size,
            lat_range=lat_range,
            min_valid_observations=min_valid_observations,
            stride=stride,
            use_preprocessed=use_preprocessed,
            include_mask_channel=include_mask_channel,
            expand_channels=expand_channels,
        )
    elif dataset_type.lower() == "era5":
        # ERA5 per-frame NPZ dataset (CI8x8 tokenizer)
        from training.era5_dataset import ERA5Dataset
        base_dataset = ERA5Dataset(
            data_pattern=data_pattern,
            crop_size=crop_size,
            random_crop=is_train,
            flip_augment=is_train,
        )
    elif dataset_type.lower() == "meso":
        # CMA-MESO NPZ dataset (CV4x8x8 tokenizer)
        from training.meso_dataset import MESODataset
        base_dataset = MESODataset(
            data_pattern=data_pattern,
            num_video_frames=num_video_frames,
            crop_size=crop_size,
            random_crop=is_train,
            flip_augment=is_train,
        )
    elif dataset_type.lower() == "video":
        # Video dataset
        if data_pattern is None:
            data_pattern = _VIDEO_PATTERN_DICT.get(dataset_name.lower(), "datasets/hdvila/videos/*.mp4")
        
        base_dataset = Dataset(
                video_pattern=data_pattern,
            num_video_frames=num_video_frames,
        )
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported. Use 'video', 'gmi', 'era5', or 'meso'.")

    # Pick the training or validation augmentations
    if is_train:
        aug_dict = video_train_augmentations(
            input_keys=["video"],  # adjust if necessary
            resolution=resolution,
            crop_height=crop_height,
        )
    else:
        aug_dict = video_val_augmentations(
            input_keys=["video"],
            resolution=resolution,
            crop_height=crop_height,
        )

    # Create channel adapter if target_channels is specified
    channel_adapter = None
    if target_channels is not None and target_channels > 0:
        channel_adapter = ChannelAdapter(
            target_channels=target_channels,
            padding_mode=channel_padding_mode,
            include_mask_channel=include_mask_channel,
        )
        mask_info = " (including mask channel)" if include_mask_channel else ""
        print(f"[Channel Adaptation] Target channels: {target_channels}{mask_info}, Padding mode: {channel_padding_mode}")

    # Wrap the dataset with the augmentations and channel adapter
    return AugmentDataset(base_dataset, aug_dict, channel_adapter=channel_adapter)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="hdvila_video")
    parser.add_argument("--dataset_type", default="video")
    parser.add_argument("--is_train", action="store_true")
    parser.add_argument("--resolution", default="720")
    parser.add_argument("--crop_height", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=25)
    args = parser.parse_args()

    dataset = dataset_entry(
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        is_train=args.is_train,
        resolution=args.resolution,
        crop_height=args.crop_height,
        num_video_frames=args.num_frames,
    )

    print(f"Total samples: {len(dataset)}")

    # 3) Grab one sample (or a few) to check shapes, keys, etc.
    if len(dataset) > 0:
        sample_idx = 0
        sample = dataset[sample_idx]
        print(f"Sample index {sample_idx} keys: {list(sample.keys())}")
        if "video" in sample:
            print("Video shape:", sample["video"].shape)
        if "video_name" in sample:
            print("Video metadata:", sample["video_name"])
        print("---\nSample loaded successfully.\n")
    else:
        print("Dataset has no samples!")
