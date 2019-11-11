#!/bin/bash
set -e
# first step is to update iblscripts
cd ~/Documents/PYTHON/iblscripts
git fetch --all
git checkout -f master
git reset --hard
git pull
# second step is to update ibllib
source ~/Documents/PYTHON/envs/iblenv/bin/activate
pip install --upgrade ibllib
pip uninstall -y phylib
pip install git+https://github.com/cortex-lab/phylib.git@master
pip install -U mtscomp