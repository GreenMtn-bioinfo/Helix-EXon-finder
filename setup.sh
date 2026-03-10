#!/bin/bash


# Check what the user provided as the GPU qualifier string (first argument)
GPU_TYPE=$(echo "$1" | tr '[:lower:]' '[:upper:]') # convert to uppercase
if [ "$GPU_TYPE" != "AMD" ]; then
    GPU_TYPE="NVIDIA"
fi


echo "Building the HEX-finder conda environment (for $GPU_TYPE hardware)..."

if [ "$GPU_TYPE" == "AMD" ]; then
    conda env create -f environment-amd.yml
else
    conda env create -f environment.yml
fi


echo "Patching $GPU_TYPE GPU library paths and variables..."

# Initialize Conda for this subshell
eval "$(conda shell.bash hook)"

# Activate the environment, permanently map the C++ libraries (for NVIDIA), and ensure TensorFlow does not use more than the first GPU
conda activate HEX-finder

if [ "$GPU_TYPE" == "AMD" ]; then
    conda env config vars set LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH" HIP_VISIBLE_DEVICES="0" ROCR_VISIBLE_DEVICES="0" 
else
    conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" CUDA_VISIBLE_DEVICES="0"
fi


echo "Setup complete for $GPU_TYPE-compatible environment! Run 'conda activate HEX-finder' to get started."