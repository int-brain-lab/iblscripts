#!/bin/bash

# Go to crontab dir
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab

last_update=$SECONDS
while true; do
  # Every four hours (or after service restart) check if any packages in the environments need updating
  elapsed=$(( SECONDS - last_update ))
  if  (( $elapsed > 14400 )); then
    echo "Checking if environments need updating"
    ../dlc/update_dlcenv.sh
    ../kilosort2/update_pykilosort.sh
    last_update=$SECONDS
  fi
  # Python: query for waiting jobs and run first job in the queue
  echo "Grabbing next large job from the queue"
  python large_jobs.py
  # Repeat
done

