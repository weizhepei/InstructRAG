#!/bin/bash

conda env create -f environment.yml

conda activate instrag

# Install flash-attn package
pip install flash-attn==2.5.6 --no-build-isolation