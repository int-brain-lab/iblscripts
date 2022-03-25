#!/bin/sh

# Make sure iblvideo is up to date
cd ~/Documents/PYTHON/iblvideo
git checkout -f master -q
git reset --hard -q
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "@{u}")
if [ $LOCAL != $REMOTE ]; then
  print("Updating iblvideo")
  git pull
fi

# Check if any pip libraries are out of date
source ~/Documents/PYTHON/envs/dlcenv/bin/activate
pip uninstall -y ibllib
pip install git+https://github.com/int-brain-lab/ibllib.git@master
pip install -U tensorflow
pip install -U deeplabcut
