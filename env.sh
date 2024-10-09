#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh

# Script to set up the environment for conda

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "conda is not installed. Please install conda and try again."
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Please install pip and try again."
    exit 1
fi

# Prompt user for environment name
read -p "Enter environment name: " env_name

# Check if the environment already exists
if ! conda env list | grep -q "$env_name"; then
    # Create the environment if it doesn't exist
    conda create -n "$env_name" python==3.8 -y || { echo "Error in creating conda environment"; exit 1; }
fi
# Activate the environment
conda activate "$env_name" || { echo "Error in activating conda environment"; exit 1; }
pip install openmim || echo "Error in pip install openmim"

# Install required packages
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html || { echo "Error in pip install torch"; exit 1; }
# pip install openmim || { echo "Error in pip install openmim"; exit 1; }
mim install mmengine==0.7.4 || { echo "Error in mim install mmengine"; exit 1; }
mim install mmcv==2.0.0rc4 || { echo "Error in mim install mmcv"; exit 1; }
mim install mmdet==3.0.0 || { echo "Error in mim install mmdet"; exit 1; }
mim install mmdet3d==1.1.0 || { echo "Error in mim install mmdet3d"; exit 1; }

# Download torch_scatter if it doesn't exist
if ! ls | grep -q "torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl"; then
    wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl || { echo "Error in wget"; exit 1; }
fi

# Install torch_scatter
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl || { echo "Error in pip install torch_scatter"; exit 1; }

# Install yapf
pip install yapf==0.40.1 || { echo "Error in pip install yapf"; exit 1; }

# Check if the script was killed by the user
if [ $? -eq 137 ]; then
    echo "bash killed"
    exit 1
fi

