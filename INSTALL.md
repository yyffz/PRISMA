mkdir -p ~/local_cuda/cuda-12.4
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
chmod 777 cuda_12.4.0_550.54.14_linux.run
sh cuda_12.4.0_550.54.15_linux.run --silent --toolkit --toolkitpath="$HOME/local_cuda/cuda-12.4" --no-opengl-libs --no-man-page --no-drm

# Create the cosmos-predict1 conda environment.
conda env create --file cosmos-predict1.yaml
# Activate the cosmos-predict1 conda environment.
conda activate cosmos-predict1
# Install the dependencies.
<!-- pip install -r requirements.txt -->
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Install Transformer engine.
<!-- pip install transformer-engine[pytorch]==1.12.0 -->
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformer-engine[pytorch]
# Install Apex for full training with bfloat16.
<!-- git clone https://github.com/NVIDIA/apex -->
<!-- CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex -->

Download apex-master.zip
unzip apex-master.zip
cd apex-master
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" .
```

You can test the environment setup for post-training with
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py --training
```
