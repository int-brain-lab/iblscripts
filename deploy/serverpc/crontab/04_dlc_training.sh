#!/usr/bin/env bash
set -e
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/dlc

# first check that we have the weights in handy. If not, get them.
if [ ! -d "trainingRig-mic-2019-02-11" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
./download_weights.sh
fi
# this will break and exit if environment doesn't exist, which is what we want
source ~/Documents/PYTHON/envs/dlc/bin/activate

# set environment variables for the CUDA processing
export PATH=/usr/local/cuda-9.0/bin:$PATH;
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# python video_pipelines.py create_flags /mnt/s0/Data/Subjects/ --dry=False
# process 5 files a day by default
python video_pipelines.py dlc_training /mnt/s0/Data/Subjects/ --dry=False --count=6
