#!/bin/bash
# update the iblvideo package
cd /home/ubuntu/Documents/PYTHON/iblvideo || exit 1
git fetch --all
git checkout -f master
git reset --hard
git pull

# update the environment
source /home/ubuntu/Documents/PYTHON/envs/dlcenv/bin/activate
pip uninstall -y ibllib
pip install git+https://github.com/int-brain-lab/ibllib.git@master
pip install -U tensorflow
pip install -U deeplabcut
