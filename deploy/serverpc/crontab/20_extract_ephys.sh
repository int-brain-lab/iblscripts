#!/usr/bin/env bash
set -e
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
source ~/Documents/PYTHON/envs/iblenv/bin/activate
python ephys.py extract_ephys /mnt/s0/Data/Subjects/ --dry=False --count=15
