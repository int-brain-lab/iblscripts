#!/bin/bash
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
source ~/Documents/PYTHON/envs/iblenv/bin/activate

# Check if the environments changed, if yes, update

# Python: query for waiting jobs and run first job in the queue
python large_jobs.py
# Repeat








# Update ibllib, iblscripts, etc. and use canary branch if canary_branch file present
./update_ibllib.sh
# Update DLC environment also now as it can otherwise lead to running DLC on partially installed env
echo "Updating DLC environment"
../dlc/update_dlcenv.sh

# Run large jobs (e.g. spike sorting, DLC) that are set to Waiting in Alyx
# this doesn't require killing as existence of the process is checked in the python code
python jobs.py run_large /mnt/s0/Data/Subjects &


echo "Updating Spike Sorting environment"
../kilosort2/update_pykilosort.sh

