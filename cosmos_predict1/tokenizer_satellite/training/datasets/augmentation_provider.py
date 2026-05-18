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

"""Augmentations for tokenizer training (image and video)"""


from cosmos_predict1.tokenizer.training.datasets.augmentors import (
    CenterCrop,
    CropResizeAugmentor,
    HorizontalFlip,
    Normalize,
    RandomReverse,
    ReflectionPadding,
    ResizeSmallestSideAspectPreserving,
    UnsqueezeImage,
)
from cosmos_predict1.tokenizer.training.datasets.utils import (
    VIDEO_KEY,
    VIDEO_RES_SIZE_INFO,
    VIDEO_VAL_CROP_SIZE_INFO,
    get_crop_size_info,
)
from cosmos_predict1.utils import log
from cosmos_predict1.utils.lazy_config import LazyCall, LazyDict

_PROB_OF_CROP_ONLY: float = 0.1


def video_train_augmentations(
    input_keys: list[str],
    resolution: str = "1080",
    crop_height: int = 256,
) -> dict[str, LazyDict]:
    """Training augmentations with all augmentation disabled, only normalization kept."""
    [_video_key] = input_keys
    log.info(f"[video] training augmentations DISABLED - only normalization will be applied.")
    # Only keep normalization, remove all augmentation operations
    augmentations = {
        "normalize": LazyCall(Normalize)(
            input_keys=[_video_key],
            output_keys=[VIDEO_KEY],
            args={"mean": 0.5, "std": 0.5},
        ),
    }

    return augmentations


def video_val_augmentations(
    input_keys: list[str], resolution: str = "1080", crop_height: int = None
) -> dict[str, LazyDict]:
    [_video_key] = input_keys
    if crop_height is None:
        crop_sizes = VIDEO_VAL_CROP_SIZE_INFO[resolution]
    else:
        crop_sizes = get_crop_size_info(crop_height)

    log.info(f"[video] validation crop_sizes: {crop_sizes}.")
    augmenations = {
        "resize_smallest_side_aspect_ratio_preserving": LazyCall(ResizeSmallestSideAspectPreserving)(
            input_keys=[VIDEO_KEY],
            args={"size": VIDEO_RES_SIZE_INFO[resolution]},
        ),
        "center_crop": LazyCall(CenterCrop)(
            input_keys=[VIDEO_KEY],
            args={"size": crop_sizes},
        ),
        "reflection_padding": LazyCall(ReflectionPadding)(
            input_keys=[VIDEO_KEY],
            args={"size": crop_sizes},
        ),
        "normalize": LazyCall(Normalize)(
            input_keys=[VIDEO_KEY],
            args={"mean": 0.5, "std": 0.5},
        ),
        "unsqueeze_padding": LazyCall(UnsqueezeImage)(input_keys=["padding_mask"]),
    }
    return augmenations
