#!/bin/bash

# Go to crontab dir
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
# Source iblenv
source ~/Documents/PYTHON/envs/iblenv/bin/activate


last_update=$SECONDS
elapsed=22000
while true; do
  # Every six hours (or after service restart) check if any packages in the environments need updating
  if  (( $elapsed > 21600 )); then
    printf "\nChecking iblenv for updates\n"
    ./update_iblenv.sh
    last_update=$SECONDS
  fi
  # Python: query for waiting jobs and run first job in the queue
  printf "\nGrabbing next small job from the queue\n"
  python small_jobs.py
  # Repeat
  elapsed=$(( SECONDS - last_update ))
done