#!/bin/bash

# Go to crontab dir
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
# Source dlcenv here. While the dlc and spike sorting tasks have their own environments, the compression jobs dont
# We avoid using iblenv here, as we don't want to interfere with the small jobs etc. dlcenv has everything needed
# for the video compression

source ~/Documents/PYTHON/envs/dlcenv/bin/activate

# Set cuda env
export CUDA_VERSION=11.8
test -e /usr/local/cuda-$CUDA_VERSION/bin || export CUDA_VERSION=11.2
export PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:/usr/local/cuda-$CUDA_VERSION/extras/CUPTI/lib64:$LD_LIBRARY_PATH

last_update=$SECONDS
elapsed=22000
while true; do
  # Every six hours (or after service restart) check if any packages in the environments need updating
  if  (( $elapsed > 21600 )); then
    printf "\nChecking dlcenv for updates\n"
    ../dlc/update_dlcenv.sh
    printf "\nChecking pyks2 env for updates\n"
    ../kilosort2/update_pykilosort.sh
    last_update=$SECONDS
  fi
  # Python: query for waiting jobs and run first job in the queue
  printf "\nGrabbing next large job from the queue\n"
  python large_jobs.py
  # Repeat
  elapsed=$(( SECONDS - last_update ))
done

