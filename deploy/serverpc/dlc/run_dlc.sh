#!/bin/bash
set -e

export CUDA_VERSION=11.8
test -e /usr/local/cuda-$CUDA_VERSION/bin || export CUDA_VERSION=11.2

export PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:/usr/local/cuda-$CUDA_VERSION/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export TF_FORCE_GPU_ALLOW_GROWTH='true'

source $1
python `dirname "$0"`/run_dlc.py $2 $3