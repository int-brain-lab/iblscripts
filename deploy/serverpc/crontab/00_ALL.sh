#!/bin/bash
set -e
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
source ~/Documents/PYTHON/envs/iblenv/bin/activate

./update_ibllib.sh

python jobs.py kill create
python jobs.py create /mnt/s0/Data/Subjects &

python jobs.py kill run
python jobs.py run /mnt/s0/Data/Subjects &

python jobs.py kill report
python jobs.py report &
