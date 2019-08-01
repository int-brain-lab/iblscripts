#!/usr/bin/env bash
source ~/Documents/PYTHON/envs/iblenv/bin/activate
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/ephys
python ephys.py extract /mnt/s0/Data/Subjects/ --dry=False --count=15
