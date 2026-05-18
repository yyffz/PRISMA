# # Download metadata with video urls
# mkdir -p datasets/hdvila
# cd datasets/hdvila
# wget https://huggingface.co/datasets/TempoFunk/hdvila-100M/resolve/main/hdvila-100M.jsonl


# The script will downlaod the original HD-VILA-100M videos, save the corresponding clips and the metadata.
python3 -m scripts.download_tokenizer_example_data --dataset_path datasets/hdvila --N_videos 128 --do_download --do_clip

# export OUTPUT_ROOT=checkpoints # default value
# export TORCH_HOME=/public/home/sunhaofei/.cache/torch

# torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
#     --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
#     experiment=Cosmos_Tokenize1_CV8x8x8_720p_HDVILA