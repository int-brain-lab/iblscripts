#!/usr/bin/env bash
source ~/Documents/PYTHON/envs/iblenv/bin/activate
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
python ephys.py raw_ephys_qc /mnt/s0/Data/Subjects/ --dry=False --count=5
