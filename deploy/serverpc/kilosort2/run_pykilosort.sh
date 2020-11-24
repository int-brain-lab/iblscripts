#!/bin/bash
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:/lib/nccl/cuda-10.0:$LD_LIBRARY_PATH; echo /usr/local/cuda-10.0
source ~/Documents/PYTHON/envs/pyks/bin/activate
#python --version
python ~/Documents/PYTHON/00_IBL/iblscripts/deploy/serverpc/kilosort2/run_pykilosort.py $1
