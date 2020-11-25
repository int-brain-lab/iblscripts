#!/bin/bash
set -e

# $1: commit or branch to test out for ibllib

# first step is to update iblscripts
cd /home/ibladmin/Documents/CI/iblscripts
git fetch --all
git reset --hard HEAD
git checkout develop
git pull --strategy-option=theirs

# second step is to update ibllib
source ~/Documents/PYTHON/envs/iblenv-ci/bin/activate

pip uninstall -y ibllib
pip uninstall -y phylib
pip install git+https://github.com/cortex-lab/phylib.git@ibl_tests
pip install git+https://github.com/int-brain-lab/ibllib.git@$1
pip install -e /home/ibladmin/Documents/CI/iblscripts

# python /home/ibladmin/Documents/CI/iblscripts/ci/download_data.py
