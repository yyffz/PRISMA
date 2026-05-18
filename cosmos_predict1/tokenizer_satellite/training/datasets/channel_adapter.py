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

"""Channel adapter for multi-instrument satellite data.

This module provides functionality to adapt satellite data with different channel numbers
to a unified channel space for training a universal tokenizer.
"""

import torch
from typing import Literal


class ChannelAdapter:
    """Adapter to normalize channel numbers across different satellite instruments.
    
    This adapter pads or repeats channels to match a target channel count,
    enabling training a unified tokenizer for multiple instruments.
    """
    
    def __init__(
        self,
        target_channels: int,
        padding_mode: Literal["zero", "repeat", "mean"] = "zero",
        include_mask_channel: bool = False,
    ):
        """
        Args:
            target_channels: Target number of channels to pad/repeat to
            padding_mode: How to handle channels when input has fewer channels:
                - "zero": Pad with zeros (default)
                - "repeat": Repeat existing channels cyclically
                - "mean": Pad with mean value of existing channels
            include_mask_channel: If True, adds an observation mask channel as the last channel.
                The mask channel indicates valid observation regions (1 for valid, 0 for invalid).
                When enabled, target_channels should include this mask channel.
        """
        self.target_channels = target_channels
        self.padding_mode = padding_mode
        self.include_mask_channel = include_mask_channel
    
    def adapt(self, video: torch.Tensor, observation_mask: torch.Tensor = None) -> torch.Tensor:
        """Adapt video tensor to target channel number.
        
        Args:
            video: Input video tensor with shape:
                - [C, H, W] for images
                - [C, T, H, W] for videos
                - [B, C, T, H, W] for batched videos
            observation_mask: Optional observation mask tensor with shape:
                - [H, W] for images
                - [T, H, W] for videos
                - [1, T, H, W] for videos with batch dimension
                Values should be 1 for valid observations and 0 for invalid regions.
                If None and include_mask_channel=True, a default mask of all ones is created.
        
        Returns:
            Adapted video tensor with target_channels channels (including mask channel if enabled)
        """
        # Determine channel dimension and current channels
        if video.ndim == 3:
            # Image format: [C, H, W]
            current_channels = video.shape[0]
            channel_dim = 0
            is_cthw = False  # Not CTHW, it's CHW
            is_image = True
        elif video.ndim == 4:
            # Assume [C, T, H, W] format for video
            current_channels = video.shape[0]
            channel_dim = 0
            is_cthw = True
            is_image = False
        elif video.ndim == 5:
            # Could be [B, C, T, H, W] or [C, T, H, W, ...]
            # For now, assume [C, T, H, W] if first dim is small
            if video.shape[0] <= 20:  # Likely channels
                current_channels = video.shape[0]
                channel_dim = 0
                is_cthw = True
                is_image = False
            else:
                # Assume [B, C, T, H, W]
                current_channels = video.shape[1]
                channel_dim = 1
                is_cthw = False
                is_image = False
        else:
            raise ValueError(
                f"Unsupported video tensor shape: {video.shape}. "
                f"Expected [C, H, W] for images, [C, T, H, W] for videos, or [B, C, T, H, W] for batched videos."
            )
        
        # Handle observation mask channel
        if self.include_mask_channel:
            # Prepare mask channel
            if observation_mask is None:
                # Create default mask (all valid)
                if is_image:
                    # [C=1, H, W]
                    mask = torch.ones((1,) + video.shape[1:], dtype=video.dtype, device=video.device)
                elif is_cthw:
                    # [1, T, H, W]
                    mask = torch.ones((1,) + video.shape[1:], dtype=video.dtype, device=video.device)
                else:
                    # [B, 1, T, H, W]
                    mask = torch.ones((video.shape[0], 1) + video.shape[2:], dtype=video.dtype, device=video.device)
            else:
                # Ensure mask has correct shape
                if is_image:
                    # Image format: mask should be [H, W] -> [1, H, W]
                    if observation_mask.ndim == 2:  # [H, W]
                        mask = observation_mask.unsqueeze(0)  # [1, H, W]
                    elif observation_mask.ndim == 3 and observation_mask.shape[0] == 1:  # [1, H, W]
                        mask = observation_mask
                    else:
                        raise ValueError(f"Mask shape {observation_mask.shape} incompatible with image format {video.shape}")
                elif observation_mask.ndim == 2:  # [H, W] for single-frame video
                    if is_cthw:
                        mask = observation_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    else:
                        mask = observation_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H, W]
                elif observation_mask.ndim == 3:  # [T, H, W] or [H, W, ...]
                    if is_cthw:
                        mask = observation_mask.unsqueeze(0)  # [1, T, H, W]
                    else:
                        mask = observation_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
                elif observation_mask.ndim == 4:
                    if is_cthw:
                        mask = observation_mask  # Already [1, T, H, W]
                    else:
                        mask = observation_mask.unsqueeze(1)  # [B, 1, T, H, W]
                else:
                    mask = observation_mask
                
                # Ensure mask is in [0, 1] range
                mask = mask.clamp(0, 1)
            
            # Adjust target_channels: if mask is included, we need one less for data channels
            data_target_channels = self.target_channels - 1
        else:
            data_target_channels = self.target_channels
            mask = None
        
        # If already correct number of channels and no mask needed, return early
        if current_channels == data_target_channels and not self.include_mask_channel:
            return video
        
        # If input has more channels than needed, truncate
        if current_channels > data_target_channels:
            # If input has more channels, take first data_target_channels
            if is_image:
                video = video[:data_target_channels, ...]
            elif is_cthw:
                video = video[:data_target_channels, ...]
            else:
                video = video[:, :data_target_channels, ...]
            current_channels = data_target_channels
        
        # If input has fewer channels, pad them
        num_padding = data_target_channels - current_channels
        
        if self.padding_mode == "zero":
            padding_value = 0.0
        elif self.padding_mode == "mean":
            if is_image:
                padding_value = video.mean(dim=(1, 2), keepdim=True)
            elif is_cthw:
                padding_value = video.mean(dim=(1, 2, 3), keepdim=True)
            else:
                padding_value = video.mean(dim=(0, 2, 3, 4), keepdim=True)
        elif self.padding_mode == "repeat":
            # Repeat channels cyclically
            if is_image:
                repeat_times = (num_padding // current_channels) + 1
                repeated = video.repeat(repeat_times, 1, 1)
                repeated = repeated[:data_target_channels, ...]
            elif is_cthw:
                repeat_times = (num_padding // current_channels) + 1
                repeated = video.repeat(repeat_times, 1, 1, 1)
                repeated = repeated[:data_target_channels, ...]
            else:
                repeat_times = (num_padding // current_channels) + 1
                repeated = video.repeat(1, repeat_times, 1, 1, 1)
                repeated = repeated[:, :data_target_channels, ...]
            
            # Append mask channel if enabled
            if self.include_mask_channel and mask is not None:
                repeated = torch.cat([repeated, mask], dim=channel_dim)
            
            return repeated
        else:
            raise ValueError(f"Unknown padding_mode: {self.padding_mode}")
        
        # Create padding tensor
        if is_image:
            # [C, H, W] format
            padding_shape = (num_padding,) + video.shape[1:]
            padding = torch.zeros(
                padding_shape,
                dtype=video.dtype,
                device=video.device
            )
            if self.padding_mode == "mean":
                padding = padding + padding_value.expand_as(padding)
            video = torch.cat([video, padding], dim=channel_dim)
        elif is_cthw:
            # [C, T, H, W] format
            padding_shape = (num_padding,) + video.shape[1:]
            padding = torch.zeros(
                padding_shape,
                dtype=video.dtype,
                device=video.device
            )
            if self.padding_mode == "mean":
                padding = padding + padding_value.expand_as(padding)
            video = torch.cat([video, padding], dim=channel_dim)
        else:
            # [B, C, T, H, W] format
            padding_shape = video.shape[:1] + (num_padding,) + video.shape[3:]
            padding = torch.zeros(
                padding_shape,
                dtype=video.dtype,
                device=video.device
            )
            if self.padding_mode == "mean":
                padding = padding + padding_value.expand_as(padding)
            video = torch.cat([video, padding], dim=channel_dim)
        
        # Append mask channel if enabled
        if self.include_mask_channel and mask is not None:
            video = torch.cat([video, mask], dim=channel_dim)
        
        return video
    
    def get_original_channels(self, adapted_video: torch.Tensor, original_channels: int) -> torch.Tensor:
        """Extract original channels from adapted video.
        
        Args:
            adapted_video: Adapted video tensor with target_channels
            original_channels: Original number of channels
        
        Returns:
            Video tensor with original_channels
        """
        if adapted_video.ndim == 3:
            # Image format: [C, H, W]
            return adapted_video[:original_channels, ...]
        elif adapted_video.ndim == 4:
            # Video format: [C, T, H, W]
            return adapted_video[:original_channels, ...]
        elif adapted_video.ndim == 5:
            # Batched format: [B, C, T, H, W]
            return adapted_video[:, :original_channels, ...]
        else:
            raise ValueError(f"Unsupported video tensor shape: {adapted_video.shape}")


def get_max_channels_from_config(instruments_config: dict) -> int:
    """Get maximum channel number from instruments configuration.
    
    Args:
        instruments_config: Dictionary mapping instrument names to channel numbers
            Example: {"instrument1": 13, "instrument2": 18, "instrument3": 10}
    
    Returns:
        Maximum channel number
    """
    if not instruments_config:
        return 3  # Default to RGB
    
    max_channels = max(instruments_config.values())
    return max_channels

