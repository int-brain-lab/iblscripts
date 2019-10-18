#!/usr/bin/env bash
set -e
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/ephys
source ~/Documents/PYTHON/envs/iblenv/bin/activate
python ephys.py sync_merge /mnt/s0/Data/Subjects/ --dry=False --count=50
