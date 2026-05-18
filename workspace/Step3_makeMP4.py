import os
import imageio.v2 as imageio  # 使用 v2 接口避免弃用警告
import numpy as np
from PIL import Image


# image_folder = './PngResult/'  # 图片所在目录
# output_path = './CREF_Obs_20240808.mp4'  # 输出视频路径
# file_list = sorted([f for f in os.listdir(image_folder) if f.startswith("2024") and f.endswith(".png")])
# print(file_list)
# print(f"找到 {len(file_list)} 帧文件")

image_folder = './'  # 图片所在目录
# output_path = './CREF_Obs_20240919.mp4'  # 输出视频路径
output_path = './CREF_Obs_20240330.mp4'  # 输出视频路径
file_list = sorted([f for f in os.listdir(image_folder) if f.startswith("2024-03") and f.endswith(".png")])
print(file_list)
print(f"找到 {len(file_list)} 帧文件")


# image_folder = './'  # 图片所在目录
# output_path = './MESO_20230428.mp4'  # 输出视频路径
# file_list = sorted([f for f in os.listdir(image_folder) if f.startswith("frame") and f.endswith(".png")])
# print(file_list)
# print(f"找到 {len(file_list)} 帧文件")



# image_folder = './Tokenizer_prep'  # 图片所在目录
# output_path = './MESO_pred_20230428.mp4'  # 输出视频路径
# file_list = sorted([f for f in os.listdir(image_folder) if f.startswith("MESO_predict_frame") and f.endswith(".png")])
# print(file_list)
# print(f"找到 {len(file_list)} 帧文件")

fps = 6  # 帧率
# 显式指定 FFmpeg 写入器配置
writer = imageio.get_writer(
    output_path,
    format='FFMPEG',      # 显式指定格式
    mode='I',             # 模式设为图像序列
    fps=fps,              # 帧率
    codec='libx264',      # 编码格式
    pixelformat='yuv420p' # 确保兼容性的像素格式
)

# 逐帧写入视频
for filename in file_list:
    img_path = os.path.join(image_folder, filename)
    image = imageio.imread(img_path)
    writer.append_data(image)

# 关闭写入器
writer.close()
print(f"视频已生成: {os.path.abspath(output_path)}")