#!/bin/bash
set -e

# task_ks2_matlab /folder/to/raw/ephys/file /scratch/folder
CUDA_VERSION=10.0  # this is the CUDA version compatible with the Matlab executable
SCRATCH_DRIVE=/mnt/h0  # this is the path of the scratch SSD volume for intermediate KS2 results swapping

# --------------- DO NOT EDIT BELOW
# sets the library path for cuda

source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate pyks2
which python  # checks the version for easier debugging
export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:/usr/local/cuda-$CUDA_VERSION/extras/CUPTI/lib64:/lib/nccl/cuda-$CUDA_VERSION:$LD_LIBRARY_PATH;
python ~/Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2/run_pykilosort.py $1 $2
