#!/bin/bash

# Go to crontab dir
cd "$HOME/Documents/PYTHON/iblscripts/deploy/serverpc/crontab"
# Source dlcenv here. While the dlc and spike sorting tasks have their own environments, the compression jobs dont
# We avoid using iblenv here, as we don't want to interfere with the small jobs etc. dlcenv has everything needed
# for the video compression
dlcenv="$HOME/Documents/PYTHON/envs/dlcenv/"
litposeenv="$HOME/Documents/PYTHON/envs/litpose/"
suite2penv="$HOME/Documents/PYTHON/envs/suite2p/"
iblsortenv="$HOME/Documents/PYTHON/SPIKE_SORTING/ibl-sorter/.venv"
source "$dlcenv/bin/activate"

# Set cuda env
export CUDA_VERSION=11.8
test -e /usr/local/cuda-$CUDA_VERSION/bin || export CUDA_VERSION=11.2
export PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:/usr/local/cuda-$CUDA_VERSION/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# we set the initial last_update to 24 hours ago to force an update on first run
last_update=$(( SECONDS - 86400 ))
last_run=$(( SECONDS - 86400 ))

while true; do
  # Every six hours (or after service restart) check if any packages in the environments need updating
  if  (( $(( SECONDS - last_update )) > 43200 )); then
    printf "\nChecking dlcenv for updates\n"
    ../dlc/update_dlcenv.sh
    if [ -d "$iblsortenv" ]; then
      printf "\nChecking iblsort env for updates\n"
      ../iblsorter/update_iblsorter.sh
    fi
    # check optional suite2p env installed
    if [ -d "$suite2penv" ]; then
      printf "\nChecking suite2p env for updates\n"
      ../mesoscope/update_suite2p_env.sh
      source "$dlcenv/bin/activate"
    fi
    last_update=$SECONDS
  fi
  # Every 5 minutes we run the large jobs
  if  (( $(( SECONDS - last_run )) > 300 )); then
    last_run=$SECONDS
    printf "\nGrabbing next large job from the queue\n"
    source "$dlcenv/bin/activate"
    python large_jobs.py
    deactivate
    if [ -d "$iblsortenv" ]; then
      source "$iblsortenv/bin/activate"
      python large_jobs.py --env iblsorter
      deactivate
    fi
    # If the suite2p env is installed, switch to this to run related task if next in queue
    if [ -d "$suite2penv" ]; then
      source "$suite2penv/bin/activate"
      python large_jobs.py --env suite2p
      deactivate
    fi
    # If the litpose env is installed, switch to this to run related task if next in queue
    if [ -d "$litposeenv" ]; then
      source "$litposeenv/bin/activate"
      python large_jobs.py --env litpose
      deactivate
    fi
  fi
  # Repeat
  sleep 5
done
