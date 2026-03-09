#!/bin/bash

echo "Building the HEX-finder conda environment..."
conda env create -f environment.yml

echo "Patching GPU library paths..."

# Initialize Conda for this subshell
eval "$(conda shell.bash hook)"

# Activate the environment and permanently map the C++ libraries
conda activate HEX-finder
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Setup complete! Run 'conda activate HEX-finder' to get started."