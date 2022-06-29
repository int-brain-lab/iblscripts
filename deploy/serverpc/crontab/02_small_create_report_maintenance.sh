#!/bin/bash

# Go to crontab dir
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
# Source iblenv
source ~/Documents/PYTHON/envs/iblenv/bin/activate

env_update_in=0 # this is to make sure the environments get updated upon service restart
env_last_update=$SECONDS  # $SECONDS indicates seconds since shell was opened, which we use a zero point
while true; do
  # If more time has elapsed since update than time until next midnight since last update: do update
  if  (( SECONDS - env_last_update >= env_update_in )); then
    printf "\nChecking iblenv for updates\n"
    ./update_iblenv.sh > /var/log/ibl/update_iblenv.log 2>&1
    # Reset time to next update to time until next midnight (in seconds)
    env_update_in=$(expr `date -d "tomorrow 0" +%s` - `date -d "now" +%s`)
    # Restart counting elapsed seconds
    env_last_update=$SECONDS
  fi
  # Python: query for waiting jobs and run first job in the queue
  printf "\nGrabbing next small job from the queue\n"
  python small_jobs.py > /var/log/ibl/small_jobs.log 2>&1
  # Repeat
done

# add create
# add report
# add maintenance