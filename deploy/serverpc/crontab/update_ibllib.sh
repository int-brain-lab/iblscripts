#!/bin/bash
set -e
# first step is to update iblscripts
cd ~/Documents/PYTHON/iblscripts
git fetch --all
git checkout -f script_test
git reset --hard
git pull
# second step is to update ibllib
# canary_branch.txt is an optional text file containing the ibllib branch to be installed
FILE=/mnt/s0/Data/Subjects/.canary_branch
if [ -f "$FILE" ]; then
    echo "$FILE exists."
    ibllib_branch="$(<$FILE)"
else
    ibllib_branch="master"
fi
# one_canary_branch.txt is an optional text file containing the one branch to be installed
FILE=/mnt/s0/Data/Subjects/.one_canary_branch
if [ -f "$FILE" ]; then
    echo "$FILE exists."
    one_branch="$(<$FILE)"
else
    one_branch="ibl_prod"
fi

source ~/Documents/PYTHON/envs/iblenv/bin/activate
# pip install --upgrade ibllib
# pip install -U mtscomp
pip install --upgrade pip
pip uninstall -y ibllib
pip uninstall -y ONE-api
pip uninstall -y phylib
pip install phylib
pip install git+https://github.com/int-brain-lab/ONE.git@$one_branch
pip install git+https://github.com/int-brain-lab/ibllib.git@$ibllib_branch --upgrade-strategy eager
#pip uninstall -y ONE-api # need to remove the ONE that was installed by ibllib
#pip install git+https://github.com/int-brain-lab/ONE.git@$one_branch
pip install -U git+https://github.com/int-brain-lab/project_extraction.git
