#!/bin/bash

# Go to crontab dir
cd "$HOME/Documents/PYTHON/iblscripts/deploy/serverpc/crontab"
# Source dlcenv here. While the dlc and spike sorting tasks have their own environments, the compression jobs dont
# We avoid using iblenv here, as we don't want to interfere with the small jobs etc. dlcenv has everything needed
# for the video compression
dlcenv="$HOME/Documents/PYTHON/envs/dlcenv/"
suite2penv="$HOME/Documents/PYTHON/envs/suite2p/"
iblsortenv="$HOME/Documents/PYTHON/SPIKE_SORTING/ibl-sorter/.venv"
source "$dlcenv/bin/activate"

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
    ../iblsorter/update_iblsorter.sh
    # check optional suite2p env installed
    if [ -d "$suite2penv" ]; then
      printf "\nChecking suite2p env for updates\n"
      ../mesoscope/update_suite2p_env.sh
      source "$dlcenv/bin/activate"
    fi
    last_update=$SECONDS
  fi
  # Python: query for waiting jobs and run first job in the queue
  printf "\nGrabbing next large job from the queue\n"
  python large_jobs.py
  python large_jobs.py --env iblsorter
  # If the suite2p env is installed, switch to this to run related task if next in queue
  if [ -d "$suite2penv" ]; then
    source "$suite2penv/bin/activate"
    python large_jobs.py --env suite2p
    deactivate
    source "$dlcenv/bin/activate"
  fi
  # Repeat
  sleep 1
  elapsed=$(( SECONDS - last_update ))
done

