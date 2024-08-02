#!/bin/bash
cd "$HOME/Documents/PYTHON/iblscripts/deploy/serverpc/crontab"
iblsortenv="$HOME/Documents/PYTHON/SPIKE_SORTING/ibl-sorter/.venv"
# comment out to force subprocess
source "$iblsortenv/bin/activate" ###

# Set cuda env
export CUDA_VERSION=11.8
test -e /usr/local/cuda-$CUDA_VERSION/bin || export CUDA_VERSION=11.2
export PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:/usr/local/cuda-$CUDA_VERSION/extras/CUPTI/lib64:$LD_LIBRARY_PATH

python spikesorting.py