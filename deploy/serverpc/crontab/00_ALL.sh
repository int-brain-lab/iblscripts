#!/bin/bash
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
source ~/Documents/PYTHON/envs/iblenv/bin/activate

./update_ibllib.sh

python jobs.py kill create
python jobs.py create /mnt/s0/Data/Subjects &

python jobs.py kill run_small
python jobs.py run_small /mnt/s0/Data/Subjects &

python jobs.py kill run_large
python jobs.py run_large /mnt/s0/Data/Subjects &

python jobs.py kill report
python jobs.py report &

../kilosort2/update_pykilosort.sh
../dlc/update_dlcenv.sh

python maintenance.py
