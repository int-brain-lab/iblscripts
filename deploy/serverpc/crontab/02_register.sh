#!/usr/bin/env bash
set -e
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/training
source ~/Documents/PYTHON/envs/iblenv/bin/activate
python experimental_data.py register /mnt/s0/Data/Subjects/ --dry=False
