#!/bin/bash
# task_ks2_matlab /folder/to/raw/ephys/file /scratch/folder
CUDA_VERSION=9.0  # this is the CUDA version compatible with the Matlab executable
SCRATCH_DRIVE=/mnt/h0  # this is the path of the scratch SSD volume for intermediate KS2 results swapping
MATLAB_BIN_PATH=/usr/bin/matlab  # put the result of the "which matlab" command in a terminal if unsure

# --------------- DO NOT EDIT BELOW
# sets the library path for cuda
export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:/usr/local/cuda-$CUDA_VERSION/extras/CUPTI/lib64:/lib/nccl/cuda-$CUDA_VERSION:$LD_LIBRARY_PATH;
$MATLAB_BIN_PATH -nodesktop -nosplash -r "addpath('~/Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2'); run_ks2_ibl('$1','$1'); exit"
