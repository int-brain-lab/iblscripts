#!/bin/bash
set -e

export PATH=/usr/local/cuda-11.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export TF_FORCE_GPU_ALLOW_GROWTH='true'

source $1
python `dirname "$0"`/run_dlc.py $2 $3