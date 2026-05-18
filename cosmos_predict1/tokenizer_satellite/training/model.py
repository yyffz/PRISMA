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

"""Implements the forward op for training, validation, and inference."""

import math
from typing import Any

import torch

from cosmos_predict1.tokenizer_satellite.training.datasets.utils import IMAGE_KEY, INPUT_KEY, MASK_KEY, RECON_KEY, VIDEO_KEY
from cosmos_predict1.tokenizer_satellite.training.losses.continuous import RECON_CONSISTENCY_KEY, VIDEO_CONSISTENCY_LOSS
from cosmos_predict1.utils import ema
from cosmos_predict1.utils.lazy_config import LazyDict, instantiate
from cosmos_predict1.utils.model import Model

PREDICTION = "prediction"
EMA_PREDICTION = "ema_prediction"


class TokenizerModel(Model):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.network = instantiate(config.network)
        self.loss = instantiate(config.loss)
        self.metric = instantiate(config.metric)
        self.precision = getattr(torch, config.precision)
        if self.config.ema.enabled:
            self.ema = ema.EMAModelTracker(
                self,
                beta=self.config.ema.beta,
                torch_compile_buffer_renaming=self.config.ema.torch_compile_buffer_renaming,
            )
        self.init_input_keys()

    def init_input_keys(self):
        self.image_key = IMAGE_KEY
        self.video_key = VIDEO_KEY

    def get_input_key(self, data_batch: dict[str, torch.Tensor]) -> str:
        if self.image_key in data_batch:
            return self.image_key
        elif self.video_key in data_batch:
            return self.video_key
        else:
            raise ValueError("Input key not found in data_batch.")

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the network.

        Args:
            optimizer_config: The optimizer config for the net.
            scheduler_config: The scheduler config for the net.

        Returns:
            optimizer (torch.optim.Optimizer): The net optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The net optimization scheduler.
        """
        optimizer_config.params = self.network.parameters()
        optimizer = instantiate(optimizer_config)
        scheduler_config.optimizer = optimizer
        scheduler = instantiate(scheduler_config)

        return optimizer, scheduler

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.ema.to(dtype=torch.float32)
        self.network = self.network.to(dtype=self.precision, memory_format=memory_format)
        self.loss = self.loss.to(dtype=self.precision, memory_format=memory_format)

    def state_dict(
        self, destination: dict[str, Any] = None, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Any]:
        original_state_dict = super(TokenizerModel, self).state_dict(destination, prefix, keep_vars)

        # Filter out '.loss' and 'ema.loss-' keys from the state dict.
        filtered_state_dict = {k: v for k, v in original_state_dict.items() if not k.startswith("loss.")}
        filtered_state_dict = {k: v for k, v in filtered_state_dict.items() if not k.startswith("ema.loss-")}
        filtered_state_dict = {
            k: v for k, v in filtered_state_dict.items() if not k.startswith("network.encoder.patcher")
        }
        filtered_state_dict = {
            k: v for k, v in filtered_state_dict.items() if not k.startswith("network.decoder.unpatcher")
        }

        return filtered_state_dict

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> None:
        own_state = self.state_dict()

        # Skip encoder input layer and decoder output layer if channel numbers differ
        # This allows loading pretrained weights when in_channels/out_channels change
        # Handle both regular model and EMA model (EMA uses '-' instead of '.' as separator)
        # Search in the original state_dict, not filtered_state_dict, to catch all keys
        encoder_conv_in_keys = [
            k for k in state_dict.keys() 
            if ("encoder.conv_in" in k or "network.encoder.conv_in" in k or 
                "encoder-conv_in" in k or "network-encoder-conv_in" in k or
                "ema.network-encoder-conv_in" in k)
        ]
        decoder_conv_out_keys = [
            k for k in state_dict.keys() 
            if ("decoder.conv_out" in k or "network.decoder.conv_out" in k or 
                "decoder-conv_out" in k or "network-decoder-conv_out" in k or
                "ema.network-decoder-conv_out" in k)
        ]
        
        # Filter state_dict: keep only keys in own_state. If a tensor has the
        # same semantic role but a changed channel shape, adapt it from the
        # pretrained tensor instead of dropping it.
        filtered_state_dict = {}
        adapted_keys = []
        skipped_shape_keys = []
        for k, v in state_dict.items():
            if k not in own_state:
                continue
            if hasattr(v, "shape") and hasattr(own_state[k], "shape") and v.shape != own_state[k].shape:
                adapted = self._adapt_pretrained_tensor(k, v, own_state[k])
                if adapted is None:
                    skipped_shape_keys.append((k, tuple(v.shape), tuple(own_state[k].shape)))
                    continue
                filtered_state_dict[k] = adapted
                adapted_keys.append((k, tuple(v.shape), tuple(own_state[k].shape)))
            else:
                filtered_state_dict[k] = v

        if adapted_keys:
            print("[Checkpoint Adaptation] Adapted pretrained tensors with changed shapes:")
            for key, old_shape, new_shape in adapted_keys:
                print(f"  - {key}: {old_shape} -> {new_shape}")
        if skipped_shape_keys:
            print("[Checkpoint Adaptation] Skipped tensors with incompatible changed shapes:")
            for key, old_shape, new_shape in skipped_shape_keys:
                print(f"  - {key}: {old_shape} -> {new_shape}")
        
        # Load only filtered state dict.
        super(TokenizerModel, self).load_state_dict(filtered_state_dict, strict=False)
        
        # Initialize encoder input layer and decoder output layer for new channels
        # Handle both regular model and EMA model
        self._initialize_input_output_layers(state_dict, encoder_conv_in_keys, decoder_conv_out_keys)
        
        # Also initialize EMA model layers if EMA is enabled
        if self.config.ema.enabled and hasattr(self, 'ema') and self.ema is not None:
            self._initialize_ema_layers(state_dict, encoder_conv_in_keys, decoder_conv_out_keys)

        # If strict is True, ensure all parameters are loaded (except the excluded ones)
        # Exclude encoder/decoder conv layers from missing keys check since we initialize them separately
        excluded_keys = set(encoder_conv_in_keys + decoder_conv_out_keys)
        # Also check for these keys in own_state (they might have different naming)
        own_excluded_keys = {
            k for k in own_state.keys()
            if ("encoder.conv_in" in k or "network.encoder.conv_in" in k or 
                "encoder-conv_in" in k or "network-encoder-conv_in" in k or
                "decoder.conv_out" in k or "network.decoder.conv_out" in k or 
                "decoder-conv_out" in k or "network-decoder-conv_out" in k or
                "ema.network-encoder-conv_in" in k or "ema.network-decoder-conv_out" in k)
        }
        excluded_keys.update(own_excluded_keys)
        
        missing_keys = set(own_state.keys()) - set(filtered_state_dict.keys()) - excluded_keys
        if missing_keys and strict:
            raise KeyError(f"Missing keys in state_dict: {missing_keys}")

    def _adapt_pretrained_tensor(self, key: str, pretrained: torch.Tensor, current: torch.Tensor) -> torch.Tensor | None:
        """Adapt a checkpoint tensor to the current tensor shape by channel-wise reuse.

        This supports widening continuous-tokenizer bottlenecks such as
        latent/z channels 16 -> 32, and also input/output channel changes.
        With checkpoint.channel_init_strategy="all_pretrained", expanded
        dimensions are filled by cycling pretrained values. With "first_only",
        only the overlapping slice is copied and the remaining initialized
        model values are kept.
        """
        if not isinstance(pretrained, torch.Tensor) or not isinstance(current, torch.Tensor):
            return None
        if pretrained.ndim != current.ndim:
            return None

        strategy = self._channel_init_strategy()
        target_shape = tuple(current.shape)
        adapted = pretrained.to(device=current.device, dtype=current.dtype)

        if strategy == "all_pretrained":
            for dim, target_size in enumerate(target_shape):
                source_size = adapted.shape[dim]
                if source_size == target_size:
                    continue
                if source_size <= 0:
                    return None
                index = torch.arange(target_size, device=adapted.device) % source_size
                adapted = adapted.index_select(dim, index)
            return adapted

        adapted_current = current.detach().clone()
        source_slices = tuple(slice(0, min(src, dst)) for src, dst in zip(pretrained.shape, target_shape))
        target_slices = tuple(slice(0, min(src, dst)) for src, dst in zip(pretrained.shape, target_shape))
        adapted_current[target_slices] = adapted[source_slices]
        return adapted_current

    def _channel_init_strategy(self) -> str:
        return getattr(
            self.config,
            "channel_init_strategy",
            getattr(getattr(self.config, "checkpoint", None), "channel_init_strategy", "first_only"),
        )

    def _initialize_input_output_layers(
        self, 
        pretrained_state_dict: dict, 
        encoder_keys: list, 
        decoder_keys: list
    ) -> None:
        """Initialize encoder input and decoder output layers when channel numbers differ.
        
        For new channels beyond pretrained channels, we can either:
        - "first_only": Only use pretrained weights for first N channels, random init for additional channels
        - "all_pretrained": Use pretrained weights for all channels (cycle through pretrained channels)
        """
        import torch.nn as nn
        import math
        
        # Get channel initialization strategy from config
        # Default to "first_only" if not specified
        channel_init_strategy = self._channel_init_strategy()
        
        # Initialize encoder input layer (first conv in Sequential)
        if encoder_keys:
            encoder = self.network.encoder if hasattr(self.network, 'encoder') else None
            if encoder and hasattr(encoder, 'conv_in'):
                # Find the first conv layer weight key (e.g., "network.encoder.conv_in.0.weight")
                weight_key = next((k for k in encoder_keys if k.endswith('.0.weight')), None)
                bias_key = next((k for k in encoder_keys if k.endswith('.0.bias')), None)
                
                if weight_key and weight_key in pretrained_state_dict:
                    pretrained_weight = pretrained_state_dict[weight_key]
                    current_weight = encoder.conv_in[0].weight.data
                    
                    # Detect if Conv2d (4D) or Conv3d (5D) based on weight dimensions
                    is_conv2d = len(pretrained_weight.shape) == 4
                    
                    # Get channel numbers (input channels are at dim=1)
                    pretrained_in_channels = pretrained_weight.shape[1]
                    current_in_channels = current_weight.shape[1]
                    
                    if current_in_channels > pretrained_in_channels:
                        # Copy pretrained channels to new model
                        if is_conv2d:
                            # Conv2d: [out_channels, in_channels, H, W]
                            current_weight[:, :pretrained_in_channels, :, :] = pretrained_weight
                            # Initialize additional channels based on strategy
                            if channel_init_strategy == "all_pretrained":
                                # Cycle through pretrained weights for additional channels
                                num_additional = current_in_channels - pretrained_in_channels
                                for i in range(num_additional):
                                    src_channel = i % pretrained_in_channels
                                    current_weight[:, pretrained_in_channels + i, :, :] = pretrained_weight[:, src_channel, :, :]
                            else:  # "first_only" or default
                                # Initialize additional channels with random values
                                nn.init.kaiming_uniform_(
                                    current_weight[:, pretrained_in_channels:, :, :], 
                                    a=math.sqrt(5)
                                )
                        else:
                            # Conv3d: [out_channels, in_channels, T, H, W]
                            current_weight[:, :pretrained_in_channels, :, :, :] = pretrained_weight
                            # Initialize additional channels based on strategy
                            if channel_init_strategy == "all_pretrained":
                                # Cycle through pretrained weights for additional channels
                                num_additional = current_in_channels - pretrained_in_channels
                                for i in range(num_additional):
                                    src_channel = i % pretrained_in_channels
                                    current_weight[:, pretrained_in_channels + i, :, :, :] = pretrained_weight[:, src_channel, :, :, :]
                            else:  # "first_only" or default
                                # Initialize additional channels with random values
                                nn.init.kaiming_uniform_(
                                    current_weight[:, pretrained_in_channels:, :, :, :], 
                                    a=math.sqrt(5)
                                )
                        strategy_msg = "pretrained weights (cycled)" if channel_init_strategy == "all_pretrained" else "random initialization"
                        print(f"[Channel Adaptation] Initialized encoder input layer: {pretrained_in_channels} -> {current_in_channels} channels ({'Conv2d' if is_conv2d else 'Conv3d'}), additional channels use {strategy_msg}")
                    elif current_in_channels == pretrained_in_channels:
                        # Same channels, just copy
                        current_weight.copy_(pretrained_weight)
                    
                    # Copy bias if exists
                    if bias_key and bias_key in pretrained_state_dict:
                        encoder.conv_in[0].bias.data.copy_(pretrained_state_dict[bias_key])
        
        # Initialize decoder output layer (first conv in Sequential)
        if decoder_keys:
            decoder = self.network.decoder if hasattr(self.network, 'decoder') else None
            if decoder and hasattr(decoder, 'conv_out'):
                # Find the first conv layer weight key (e.g., "network.decoder.conv_out.0.weight")
                weight_key = next((k for k in decoder_keys if k.endswith('.0.weight')), None)
                bias_key = next((k for k in decoder_keys if k.endswith('.0.bias')), None)
                
                if weight_key and weight_key in pretrained_state_dict:
                    pretrained_weight = pretrained_state_dict[weight_key]
                    current_weight = decoder.conv_out[0].weight.data
                    
                    # Detect if Conv2d (4D) or Conv3d (5D) based on weight dimensions
                    is_conv2d = len(pretrained_weight.shape) == 4
                    
                    # Get channel numbers (output channels are at dim=0)
                    pretrained_out_channels = pretrained_weight.shape[0]
                    current_out_channels = current_weight.shape[0]
                    
                    if current_out_channels > pretrained_out_channels:
                        # Copy pretrained channels to new model
                        if is_conv2d:
                            # Conv2d: [out_channels, in_channels, H, W]
                            current_weight[:pretrained_out_channels, :, :, :] = pretrained_weight
                            # Initialize additional channels based on strategy
                            if channel_init_strategy == "all_pretrained":
                                # Cycle through pretrained weights for additional channels
                                num_additional = current_out_channels - pretrained_out_channels
                                for i in range(num_additional):
                                    src_channel = i % pretrained_out_channels
                                    current_weight[pretrained_out_channels + i, :, :, :] = pretrained_weight[src_channel, :, :, :]
                            else:  # "first_only" or default
                                # Initialize additional channels with random values
                                nn.init.kaiming_uniform_(
                                    current_weight[pretrained_out_channels:, :, :, :], 
                                    a=math.sqrt(5)
                                )
                        else:
                            # Conv3d: [out_channels, in_channels, T, H, W]
                            current_weight[:pretrained_out_channels, :, :, :, :] = pretrained_weight
                            # Initialize additional channels based on strategy
                            if channel_init_strategy == "all_pretrained":
                                # Cycle through pretrained weights for additional channels
                                num_additional = current_out_channels - pretrained_out_channels
                                for i in range(num_additional):
                                    src_channel = i % pretrained_out_channels
                                    current_weight[pretrained_out_channels + i, :, :, :, :] = pretrained_weight[src_channel, :, :, :, :]
                            else:  # "first_only" or default
                                # Initialize additional channels with random values
                                nn.init.kaiming_uniform_(
                                    current_weight[pretrained_out_channels:, :, :, :, :], 
                                    a=math.sqrt(5)
                                )
                        strategy_msg = "pretrained weights (cycled)" if channel_init_strategy == "all_pretrained" else "random initialization"
                        print(f"[Channel Adaptation] Initialized decoder output layer: {pretrained_out_channels} -> {current_out_channels} channels ({'Conv2d' if is_conv2d else 'Conv3d'}), additional channels use {strategy_msg}")
                    elif current_out_channels == pretrained_out_channels:
                        # Same channels, just copy
                        current_weight.copy_(pretrained_weight)
                    
                    # Copy bias if exists
                    if bias_key and bias_key in pretrained_state_dict:
                        decoder.conv_out[0].bias.data.copy_(pretrained_state_dict[bias_key])

    def _initialize_ema_layers(
        self, 
        pretrained_state_dict: dict, 
        encoder_keys: list, 
        decoder_keys: list
    ) -> None:
        """Initialize EMA model's encoder input and decoder output layers when channel numbers differ.
        
        EMA model uses '-' as separator instead of '.' in state_dict keys.
        EMA weights are stored as buffers in self.ema.
        """
        import torch.nn as nn
        import math
        from cosmos_predict1.utils.ema import get_buffer_name
        
        # Get channel initialization strategy from config (same as main model)
        channel_init_strategy = self._channel_init_strategy()
        
        if not hasattr(self, 'ema') or self.ema is None:
            return
        
        # Get the main model's encoder and decoder for reference
        encoder = self.network.encoder if hasattr(self.network, 'encoder') else None
        decoder = self.network.decoder if hasattr(self.network, 'decoder') else None
        
        if encoder is None or decoder is None:
            return
        
        # Initialize encoder input layer for EMA model
        # EMA keys in state_dict have format: "ema.network-encoder-conv_in-0-weight" or "ema.network-encoder-conv_in-weight"
        ema_encoder_keys = [k for k in encoder_keys if k.startswith("ema.")]
        if ema_encoder_keys and encoder and hasattr(encoder, 'conv_in'):
            # Find EMA encoder weight key (try both with and without "0")
            weight_key = next((k for k in ema_encoder_keys if 'weight' in k), None)
            bias_key = next((k for k in ema_encoder_keys if 'bias' in k), None)
            
            if weight_key and weight_key in pretrained_state_dict:
                pretrained_weight = pretrained_state_dict[weight_key]
                # Try to find the corresponding buffer name in EMA tracker
                # Parameter name: "network.encoder.conv_in.0.weight" -> Buffer name: "network-encoder-conv_in-0-weight"
                param_name = "network.encoder.conv_in.0.weight"
                buffer_name = get_buffer_name(param_name, self.ema.torch_compile_buffer_renaming)
                
                # If buffer doesn't exist, try without "0" (some checkpoints may use different format)
                if not hasattr(self.ema, buffer_name):
                    # Try alternative format: "network-encoder-conv_in-weight" (without "0")
                    alt_buffer_name = buffer_name.replace("-0-weight", "-weight")
                    if hasattr(self.ema, alt_buffer_name):
                        buffer_name = alt_buffer_name
                
                if hasattr(self.ema, buffer_name):
                    current_buffer = getattr(self.ema, buffer_name)
                    
                    # Detect if Conv2d (4D) or Conv3d (5D) based on weight dimensions
                    is_conv2d = len(pretrained_weight.shape) == 4
                    
                    pretrained_in_channels = pretrained_weight.shape[1]
                    current_in_channels = current_buffer.shape[1]
                    
                    if current_in_channels > pretrained_in_channels:
                        if is_conv2d:
                            # Conv2d: [out_channels, in_channels, H, W]
                            current_buffer[:, :pretrained_in_channels, :, :] = pretrained_weight
                            # Initialize additional channels based on strategy
                            if channel_init_strategy == "all_pretrained":
                                # Cycle through pretrained weights for additional channels
                                num_additional = current_in_channels - pretrained_in_channels
                                for i in range(num_additional):
                                    src_channel = i % pretrained_in_channels
                                    current_buffer[:, pretrained_in_channels + i, :, :] = pretrained_weight[:, src_channel, :, :]
                            else:  # "first_only" or default
                                # Initialize additional channels with random values
                                nn.init.kaiming_uniform_(
                                    current_buffer[:, pretrained_in_channels:, :, :], 
                                    a=math.sqrt(5)
                                )
                        else:
                            # Conv3d: [out_channels, in_channels, T, H, W]
                            current_buffer[:, :pretrained_in_channels, :, :, :] = pretrained_weight
                            # Initialize additional channels based on strategy
                            if channel_init_strategy == "all_pretrained":
                                # Cycle through pretrained weights for additional channels
                                num_additional = current_in_channels - pretrained_in_channels
                                for i in range(num_additional):
                                    src_channel = i % pretrained_in_channels
                                    current_buffer[:, pretrained_in_channels + i, :, :, :] = pretrained_weight[:, src_channel, :, :, :]
                            else:  # "first_only" or default
                                # Initialize additional channels with random values
                                nn.init.kaiming_uniform_(
                                    current_buffer[:, pretrained_in_channels:, :, :, :], 
                                    a=math.sqrt(5)
                                )
                        strategy_msg = "pretrained weights (cycled)" if channel_init_strategy == "all_pretrained" else "random initialization"
                        print(f"[Channel Adaptation] Initialized EMA encoder input layer: {pretrained_in_channels} -> {current_in_channels} channels ({'Conv2d' if is_conv2d else 'Conv3d'}), additional channels use {strategy_msg}")
                    elif current_in_channels == pretrained_in_channels:
                        current_buffer.copy_(pretrained_weight)
                
                # Handle bias
                if bias_key and bias_key in pretrained_state_dict:
                    param_name_bias = "network.encoder.conv_in.0.bias"
                    buffer_name_bias = get_buffer_name(param_name_bias, self.ema.torch_compile_buffer_renaming)
                    if not hasattr(self.ema, buffer_name_bias):
                        alt_buffer_name_bias = buffer_name_bias.replace("-0-bias", "-bias")
                        if hasattr(self.ema, alt_buffer_name_bias):
                            buffer_name_bias = alt_buffer_name_bias
                    if hasattr(self.ema, buffer_name_bias):
                        getattr(self.ema, buffer_name_bias).copy_(pretrained_state_dict[bias_key])
        
        # Initialize decoder output layer for EMA model
        ema_decoder_keys = [k for k in decoder_keys if k.startswith("ema.")]
        if ema_decoder_keys and decoder and hasattr(decoder, 'conv_out'):
            # Find EMA decoder weight key
            weight_key = next((k for k in ema_decoder_keys if 'weight' in k), None)
            bias_key = next((k for k in ema_decoder_keys if 'bias' in k), None)
            
            if weight_key and weight_key in pretrained_state_dict:
                pretrained_weight = pretrained_state_dict[weight_key]
                # Parameter name: "network.decoder.conv_out.0.weight" -> Buffer name: "network-decoder-conv_out-0-weight"
                param_name = "network.decoder.conv_out.0.weight"
                buffer_name = get_buffer_name(param_name, self.ema.torch_compile_buffer_renaming)
                
                if hasattr(self.ema, buffer_name):
                    current_buffer = getattr(self.ema, buffer_name)
                    
                    # Detect if Conv2d (4D) or Conv3d (5D) based on weight dimensions
                    is_conv2d = len(pretrained_weight.shape) == 4
                    
                    pretrained_out_channels = pretrained_weight.shape[0]
                    current_out_channels = current_buffer.shape[0]
                    
                    if current_out_channels > pretrained_out_channels:
                        if is_conv2d:
                            # Conv2d: [out_channels, in_channels, H, W]
                            current_buffer[:pretrained_out_channels, :, :, :] = pretrained_weight
                            # Initialize additional channels based on strategy
                            if channel_init_strategy == "all_pretrained":
                                # Cycle through pretrained weights for additional channels
                                num_additional = current_out_channels - pretrained_out_channels
                                for i in range(num_additional):
                                    src_channel = i % pretrained_out_channels
                                    current_buffer[pretrained_out_channels + i, :, :, :] = pretrained_weight[src_channel, :, :, :]
                            else:  # "first_only" or default
                                # Initialize additional channels with random values
                                nn.init.kaiming_uniform_(
                                    current_buffer[pretrained_out_channels:, :, :, :], 
                                    a=math.sqrt(5)
                                )
                        else:
                            # Conv3d: [out_channels, in_channels, T, H, W]
                            current_buffer[:pretrained_out_channels, :, :, :, :] = pretrained_weight
                            # Initialize additional channels based on strategy
                            if channel_init_strategy == "all_pretrained":
                                # Cycle through pretrained weights for additional channels
                                num_additional = current_out_channels - pretrained_out_channels
                                for i in range(num_additional):
                                    src_channel = i % pretrained_out_channels
                                    current_buffer[pretrained_out_channels + i, :, :, :, :] = pretrained_weight[src_channel, :, :, :, :]
                            else:  # "first_only" or default
                                # Initialize additional channels with random values
                                nn.init.kaiming_uniform_(
                                    current_buffer[pretrained_out_channels:, :, :, :, :], 
                                    a=math.sqrt(5)
                                )
                        strategy_msg = "pretrained weights (cycled)" if channel_init_strategy == "all_pretrained" else "random initialization"
                        print(f"[Channel Adaptation] Initialized EMA decoder output layer: {pretrained_out_channels} -> {current_out_channels} channels ({'Conv2d' if is_conv2d else 'Conv3d'}), additional channels use {strategy_msg}")
                    elif current_out_channels == pretrained_out_channels:
                        current_buffer.copy_(pretrained_weight)
                
                # Handle bias
                if bias_key and bias_key in pretrained_state_dict:
                    param_name_bias = "network.decoder.conv_out.0.bias"
                    buffer_name_bias = get_buffer_name(param_name_bias, self.ema.torch_compile_buffer_renaming)
                    if hasattr(self.ema, buffer_name_bias):
                        getattr(self.ema, buffer_name_bias).copy_(pretrained_state_dict[bias_key])

    def _on_before_network_forward(self, data_batch: dict[str, torch.Tensor]) -> None:
        consistency_loss = self.loss.loss_modules[VIDEO_CONSISTENCY_LOSS]
        if hasattr(consistency_loss, "enabled") and consistency_loss.enabled:
            _input_key = self.get_input_key(data_batch)
            # print(f'_input_key: {_input_key}')
            # print(f'self.video_key: {self.video_key}')
            # print(f'data_batch[_input_key].shape: {data_batch[_input_key].shape}')
            if _input_key is self.video_key:
                data_batch[_input_key] = consistency_loss.shuffle(data_batch[_input_key])
            # print(f'data_batch[_input_key].shape: {data_batch[_input_key].shape}')
        return

    def _on_after_network_forward(
        self, data_batch: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor]
    ) -> None:
        consistency_loss = self.loss.loss_modules[VIDEO_CONSISTENCY_LOSS]
        if hasattr(consistency_loss, "enabled") and consistency_loss.enabled:
            _input_key = self.get_input_key(data_batch)
            if _input_key is self.video_key:
                # 还原输入数据
                data_batch[_input_key] = consistency_loss.unshuffle(data_batch[_input_key])
                # print(f'data_batch[_input_key].shape after unshuffle: {data_batch[_input_key].shape}')
                # 创建一致性重建结果（在unshuffle之前，用于计算一致性损失）
                # print(f'output_batch[RECON_KEY].shape: {output_batch[RECON_KEY].shape}')
                output_batch[RECON_CONSISTENCY_KEY] = torch.ones_like(output_batch[RECON_KEY]) * output_batch[RECON_KEY]
                # 还原重建结果（如果模型输出时间维度不匹配，unshuffle会直接返回）
                output_batch[RECON_KEY] = consistency_loss.unshuffle(output_batch[RECON_KEY])
        return

    def _network_forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # A callback proxy to modify the input before the forward pass.
        self._on_before_network_forward(data_batch)

        # Do the forward pass.
        tensor_batch = data_batch[self.get_input_key(data_batch)]
        output_batch = self.network(tensor_batch)
        output_batch = output_batch if self.network.training else output_batch._asdict()

        # A callback proxy to modify the output after the forward pass.
        self._on_after_network_forward(data_batch, output_batch)
        return output_batch

    def training_step(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        _input_key = self.get_input_key(data_batch)
        output_dict = self._network_forward(data_batch)
        input_images, recon_images = data_batch[_input_key], output_dict[RECON_KEY]

        # pass loss_mask to loss computation
        inputs = {INPUT_KEY: input_images, MASK_KEY: data_batch.get("loss_mask", torch.ones_like(input_images))}

        loss_dict, loss_value = self.loss(inputs, output_dict, iteration)
        return dict({PREDICTION: recon_images, **loss_dict}), loss_value

    @torch.no_grad()
    def validation_step(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
        ema_model: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        _input_key = self.get_input_key(data_batch)
        output_dict = self._network_forward(data_batch)
        input_images, recon_images = data_batch[_input_key], output_dict[RECON_KEY]

        # pass loss_mask to loss computation
        inputs = {INPUT_KEY: input_images, MASK_KEY: data_batch.get("loss_mask", torch.ones_like(input_images))}

        loss_dict, loss_value = self.loss(inputs, output_dict, iteration)
        metric_dict = self.metric(input_images, output_dict, iteration)
        loss_dict.update(metric_dict)
        prediction_key = EMA_PREDICTION if ema_model else PREDICTION
        return dict({prediction_key: recon_images, **loss_dict}), loss_value

    @torch.inference_mode()
    def forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _input_key = self.get_input_key(data_batch)
        output_dict = self._network_forward(data_batch)
        return dict({PREDICTION: output_dict[RECON_KEY]})
