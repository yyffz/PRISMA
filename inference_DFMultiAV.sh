# Download pre-trained autoregressive checkpoints
# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_autoregressive_checkpoints.py --model_sizes 4B --checkpoint_dir checkpoints

# # 预处理雷达数据
# python workspace/Step1_Bin2NC.py \
#     --data_dir /public/home/sunhaofei/cosmos-predict1/datasets/Radar_Data/202306/ \
#     --save_dir ./ \
#     --start_str 20230601000000 \
#     --end_str 20230701000000 \
#     --start_lon 114 \
#     --start_lat 28 

#     # --data_dir /public/DataWarehouse/OBS/CMAMESO/radarl3crefqc/2024/08/08/ \
 




# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/base.py \
#     --checkpoint_dir checkpoints \
#     --ar_model_dir Cosmos-Predict1-4B \
#     --input_type video \
#     --input_image_or_video_path assets/autoregressive/input.mp4 \
#     --top_p 0.8 \
#     --temperature 1.0 \
#     --offload_diffusion_decoder \
#     --offload_tokenizer \
#     --offload_guardrail_models \
#     # --offload_text_encoder_model \
#     # --offload_prompt_upsampler \
#     --video_save_name autoregressive-4b



# NUM_GPUS=4
# CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/autoregressive/inference/base.py \
#     --num_gpus ${NUM_GPUS} \
#     --checkpoint_dir checkpoints \
#     --ar_model_dir Cosmos-Predict1-4B \
#     --input_type video \
#     --input_image_or_video_path assets/autoregressive/input.mp4 \
#     --top_p 0.8 \
#     --temperature 1.0 \
#     --offload_diffusion_decoder \
#     --offload_tokenizer \
#     --offload_guardrail_models
#     --video_save_name autoregressive-4b-4gpu


# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py --help

PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. \
The video captures a nighttime drive through a suburban area. The road is illuminated by streetlights and vehicle headlights, creating a well-lit path. \
Palm trees line both sides of the street, adding a tropical feel to the environment. \
Several cars are parked along the right side of the road, and a few are driving in the opposite direction. \
The sky is overcast, with no visible stars or moon, indicating it is likely late evening or night. \
The overall scene is quiet and peaceful, with no pedestrians or significant traffic."

PROMPT_LEFT="The video is captured from a camera mounted on a car. The camera is facing to the left. \
The video captures a nighttime drive through a quiet neighborhood. A white car is seen turning left onto a street lined with palm trees and other trees. \
The street is illuminated by streetlights, and there are a few parked cars visible. \
The sky is overcast, and the overall scene is dimly lit, indicating it is nighttime."

PROMPT_RIGHT="The video is captured from a camera mounted on a car. The camera is facing to the right. \
The The video captures a nighttime driving scene on a multi-lane road. The road is bordered by a high concrete barrier. \
Several cars are visible, including a white van, a black sedan, and a red car. \
The traffic appears to be moving slowly, possibly due to congestion. The sky is overcast, and the lighting suggests it is nighttime. \
Trees and some buildings can be seen in the background, indicating an urban or suburban setting."

PROMPT_BACK="The video is captured from a camera mounted on a car. The camera is facing backwards. \
The video depicts a nighttime drive through a residential area. The street is illuminated by streetlights and the headlights of a car in front. \
The road is lined with parked cars and houses, and there are trees and a fence along the side. \
A white truck is seen turning left onto the street, and a person is standing near the fence. \
The scene is quiet and there are no visible pedestrians or other vehicles. The weather appears to be clear."

PROMPT_BACK_LEFT="The video is captured from a camera mounted on a car. The camera is facing the rear left side."

PROMPT_BACK_RIGHT="The video is captured from a camera mounted on a car. The camera is facing the rear right side."


CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World-Sample-AV-Single2MultiView/v2w_model.pt \
    --view_condition_video assets/diffusion/sv2mv_input_view_2.mp4 \
    --initial_condition_video assets/diffusion/sv2mv_initial_frames.mp4 \
    --num_input_frames 9 \
    --condition_location "first_cam_and_first_n" \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back "${PROMPT_BACK}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-single2multiview-video2world