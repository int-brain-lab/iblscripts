#!/bin/bash

# Go to crontab dir
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
# Source iblenv
source ~/Documents/PYTHON/envs/iblenv/bin/activate

env_update_in=0 # this is to make sure the environments get updated upon service restart
env_last_update=$SECONDS  # $SECONDS indicates seconds since shell was opened, which we use a zero point

report_create_elapsed=7200  # this is to make sure report and create run upon service restart
report_create_last=$SECONDS  # $SECONDS indicates seconds since shell was opened, which we use a zero point

while true; do
  # Every night at midnight, run environment update and maintenance script
  if  (( SECONDS - env_last_update >= env_update_in )); then
    printf "\n$(date)\n" ;
    printf "Checking iblenv for updates\n" ;
    printf "Logging to /var/log/ibl/update_iblenv.log\n" ;
    ./update_iblenv.sh >> /var/log/ibl/update_iblenv.log 2>&1 ;

    printf "\n$(date)\n" ;
    printf "Running maintenance script\n" ;
    printf "Logging to /var/log/ibl/maintenance_jobs.log\n" ;
    python maintenance_jobs.py >> /var/log/ibl/maintenance_jobs.log 2>&1 ;

    # Reset time to next update to time until next midnight (in seconds) and restart counting elapsed seconds
    env_update_in=$(expr `date -d "tomorrow 0" +%s` - `date -d "now" +%s`) ;
    env_last_update=$SECONDS ;
  fi

  # If more than two hours have passed, report health and check for create jobs
  if  (( SECONDS - report_create_last >= 7200 )); then
    printf "\n$(date)\n" ;
    printf "Running health report and creating jobs\n" ;
    printf "Logging to /var/log/ibl/report_create_jobs.log\n" ;
    python report_create_jobs.py >> /var/log/ibl/report_create_jobs.log 2>&1 ;
    report_create_last=$SECONDS  # reset the timer
  fi

  # Always: query for waiting jobs and run first job in the queue
  printf "\n$(date)\n" ;
  printf "Running next set of small jobs from the queue\n" ;
  printf "Logging to /var/log/ibl/small_jobs.log\n" ;
  python small_jobs.py >> /var/log/ibl/small_jobs.log 2>&1 ;

  # Repeat
done