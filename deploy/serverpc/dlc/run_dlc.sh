#!/bin/bash
set -e

export PATH=/usr/local/cuda-11.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export TF_FORCE_GPU_ALLOW_GROWTH='true'

source ~/Documents/PYTHON/envs/dlcenv/bin/activate
python ~/Documents/PYTHON/iblscripts/deploy/serverpc/dlc/run_dlc.py