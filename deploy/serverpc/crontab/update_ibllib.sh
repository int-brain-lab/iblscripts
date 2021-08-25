#!/bin/bash
set -e
# first step is to update iblscripts
cd ~/Documents/PYTHON/iblscripts
git fetch --all
git checkout -f master
git reset --hard
git pull
# second step is to update ibllib
# canary_branch.txt is an optional text file containing the ibllib branch to be installed
FILE=/mnt/s0/Data/Subjects/.canary_branch
if [ -f "$FILE" ]; then
    echo "$FILE exists."
    branch="$(<$FILE)"
else
    branch="master"
fi
source ~/Documents/PYTHON/envs/iblenv/bin/activate
# pip install --upgrade ibllib
# pip install -U mtscomp
pip install --upgrade pip
pip uninstall -y ibllib
pip uninstall -y phylib
pip install phylib
pip install git+https://github.com/int-brain-lab/ibllib.git@$branch --upgrade-strategy eager
pip install -U git+https://github.com/int-brain-lab/ONE.git

# auto-update pykilosort
cd ~/Documents/PYTHON/SPIKE_SORTING/pykilosort
git fetch --all
git checkout -f ibl_prod
git reset --hard
git pull
