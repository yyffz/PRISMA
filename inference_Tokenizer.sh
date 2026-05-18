

# python workspace/Step4_SelectCase.py  \
#     --input_dir ./datasets/Radar_SH/202307/ \
#     --output_dir ./datasets/NowcastCase_SH \



# Autoencoding videos using `Cosmos-Tokenize1-DV4x8x8-360p`.
model_name="Cosmos-Tokenize1-DV4x8x8-360p"
python3 -m cosmos_predict1.tokenizer.inference.video_cli \
    --video_pattern 'cosmos_predict1/tokenizer/test_data/video.mp4' \
    --checkpoint_enc checkpoints/${model_name}/encoder.jit \
    --checkpoint_dec checkpoints/${model_name}/decoder.jit \
    --output_dir 'cosmos_predict1/tokenizer/test_data/reconstructions/' \