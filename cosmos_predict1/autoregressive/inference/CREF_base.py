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

import argparse
import os
import numpy as np
import imageio
import torch

from cosmos_predict1.autoregressive.inference.world_generation_pipeline import ARBaseGenerationPipeline
from cosmos_predict1.autoregressive.utils.inference import add_common_arguments, load_vision_input, validate_args
from cosmos_predict1.utils import log


def parse_args():
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)
    parser.add_argument(
        "--ar_model_dir",
        type=str,
        default="Cosmos-Predict1-4B",
    )
    parser.add_argument("--input_type", type=str, default="video", help="Type of input", choices=["image", "video"])
    args = parser.parse_args()
    return args


def main(args):
    """Run video-to-world generation demo.

    This function handles the main video-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple images/videos from input
    - Generating videos from images/videos
    - Saving the generated videos to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (temperature, top_p)
            - Input/output settings (images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    inference_type = "base"  # When the inference_type is "base", AR model does not take text as input, the world generation is purely based on the input video
    sampling_config = validate_args(args, inference_type)

    if args.num_gpus > 1:
        from megatron.core import parallel_state

        from cosmos_predict1.utils import distributed

        distributed.init()

    # Initialize base generation model pipeline
    pipeline = ARBaseGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.ar_model_dir,
        disable_diffusion_decoder=args.disable_diffusion_decoder,
        # offload_guardrail_models=args.offload_guardrail_models,
        offload_guardrail_models=True,
        offload_diffusion_decoder=args.offload_diffusion_decoder,
        offload_network=args.offload_ar_model,
        offload_tokenizer=args.offload_tokenizer,
        disable_guardrail=args.disable_guardrail,
        # disable_guardrail=True,
        parallel_size=args.num_gpus,
    )

    # # Load input image(s) or video(s)
    # input_videos = load_vision_input(
    #     input_type=args.input_type,
    #     batch_input_path=args.batch_input_path,
    #     input_image_or_video_path=args.input_image_or_video_path,
    #     data_resolution=args.data_resolution,
    #     num_input_frames=args.num_input_frames,
    # )

    # input_videos = os.path.join(args.input_image_or_video_path, f"{args.video_save_name}.mp4")

    # for idx, input_filename in enumerate(input_videos):
    # inp_vid = input_videos[input_filename]

    # data = np.load("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/cosmos-predict1-main/workspace/cref_20250418T0742_window000.npz")
    # data = np.load("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/cosmos-predict1-main/workspace/cref_20250505T1000_window002.npz")
    # data = np.load("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/cosmos-predict1-main/workspace/cref_20250505T0900_window001.npz")
    # data = np.load("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/cosmos-predict1-main/workspace/cref_20250522T0800_window000.npz")
    # data = np.load("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/cosmos-predict1-main/workspace/cref_20250522T0900_window001.npz")
    
    # data = np.load("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/cosmos-predict1-main/workspace/cref_20250530T0000_window000.npz")
    # data = np.load("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/cosmos-predict1-main/workspace/cref_20250530T0100_window001.npz")
    # data = np.load("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/cosmos-predict1-main/workspace/cref_20250530T0800_window000.npz")
    # data = np.load("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/cosmos-predict1-main/workspace/cref_20250530T0900_window001.npz")
    # data = np.load("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/lizeting/Sunhaofei/cosmos-predict1-main/workspace/cref_20250530T1000_window002.npz")
    
    data = np.load("/public/home/sunhaofei/cosmos-predict1/case_0825_AR/cref_20250825T1100.npz")
    
    # print(data.files)
    inp_vid_tensor = torch.from_numpy(data['arr_0'])  # 先转换为torch.Tensor
    inp_vid = inp_vid_tensor.unsqueeze(0) 

    # Generate video
    # log.info(f"Run with image or video path: {input_filename}")
    log.info(f"Run with image or video path: {inp_vid.shape}")
    log.info(f"Run with image or video path: {type(inp_vid)}")
    out_vid = pipeline.generate(
        inp_vid=inp_vid,
        num_input_frames=args.num_input_frames,
        seed=args.seed,
        sampling_config=sampling_config,
    )
    log.info(f"Run with image or video path: {type(out_vid)}")
    log.info(f"Run with image or video path: {out_vid.shape}")
    if out_vid is None:
        log.critical("Guardrail blocked base generation.")
        # continue
    
    # save out_vid as npy
    # np.save(f"./AR_pred_08250600.npy", out_vid.cpu().numpy()) 
    out_vid_float = out_vid.cpu().to(torch.float32)  # 转换为 float32 类型
    np.save(f"./AR_pred_08251100.npy", out_vid_float.numpy())  # 现在可以正常保存

    # # Save video
    # if args.input_image_or_video_path:
    #     out_vid_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4")
    # else:
    #     out_vid_path = os.path.join(args.video_save_folder, f"{idx}.mp4")

    # imageio.mimsave(out_vid_path, out_vid, fps=25)
    # log.info(f"Saved video to {out_vid_path}")

    # # clean up properly
    # if args.num_gpus > 1:
    #     parallel_state.destroy_model_parallel()
    #     import torch.distributed as dist

    #     dist.destroy_process_group()


if __name__ == "__main__":
    torch._C._jit_set_texpr_fuser_enabled(False)
    args = parse_args()
    main(args)
