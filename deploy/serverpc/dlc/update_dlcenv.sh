#!/bin/bash

# Make sure local iblvideo repository is up to date
cd ~/Documents/PYTHON/iblvideo
git checkout -f master -q
git reset --hard -q
git fetch
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "@{u}")
if [ $LOCAL != $REMOTE ]; then
  echo "Updating iblvideo"
  git pull
else
  echo "iblvideo is up-to-date"
fi

# Check if any pip libraries are out of date if yes, update
source ~/Documents/PYTHON/envs/dlcenv/bin/activate
echo "Updating ibllib mesoscope"
pip install --upgrade git+https://github.com/int-brain-lab/ibllib.git@mesoscope --upgrade-strategy eager ;
