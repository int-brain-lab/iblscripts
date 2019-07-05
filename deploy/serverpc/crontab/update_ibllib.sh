#!/bin/bash
set -e
# first step is to update iblscripts
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
git fetch --all
git checkout master
git reset --hard origin/master
# second step is to update ibllib
source ~/Documents/PYTHON/envs/iblenv/bin/activate
pip install --upgrade ibllib
# pip install ibllib==1.0.1
