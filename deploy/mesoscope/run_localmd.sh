#!/bin/bash

source ~/Documents/PYTHON/envs/pmdenv/bin/activate
CUDA_VERSION=11.8
export PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH 
export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:/usr/local/cuda-$CUDA_VERSION/extras/CUPTI/lib64:$LD_LIBRARY_PATH

which python
nvcc --version

# $1 session_path
# $2 FOV number
# $3 block height
# $4 block width
python ~/Documents/PYTHON/iblscripts/mesoscope/run_localmd.py --session_path $1 --fov $2 --block_height $3 --block_width $4