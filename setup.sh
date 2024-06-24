#!/bin/bash

# Create a new conda environment with Python 3.10
conda create -n instrag python=3.10 -y

# Activate the new conda environment
conda activate instrag

# Install numpy, vllm, and accelerate
pip install numpy==1.26.4 vllm==0.4.1 accelerate

# Install flash-attn
pip install flash-attn==2.5.6 --no-build-isolation
