CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/MESO_base.py \
    --checkpoint_dir checkpoints \
    --ar_model_dir Cosmos-Predict1-4B \
    --input_type video \
    --input_image_or_video_path assets/autoregressive/ \
    --top_p 0.8 \
    --temperature 1.0 \
    --offload_diffusion_decoder \
    --offload_tokenizer \
    --video_save_name autoregressive-4b_MESO_20230428 


# NUM_GPUS=4
# CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/autoregressive/inference/MESO_base.py \
#     --num_gpus ${NUM_GPUS} \
#     --checkpoint_dir checkpoints \
#     --ar_model_dir Cosmos-Predict1-4B \
#     --input_type video \
#     --input_image_or_video_path assets/autoregressive/input.mp4 \
#     --top_p 0.8 \
#     --temperature 1.0 \
#     --offload_diffusion_decoder \
#     --offload_tokenizer \
#     --video_save_name autoregressive-4b_MESO_20230428 

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