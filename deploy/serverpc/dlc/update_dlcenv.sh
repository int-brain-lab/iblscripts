#!/bin/bash

# Make sure local iblvideo repository is up to date
cd ~/Documents/PYTHON/iblvideo
git checkout -f master -q
git reset --hard -q
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

outdated=$(pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1)
for lib in "ibllib" "tensorflow" "deeplabcut"
do
  update=$(echo $outdated | grep -o $lib | cut -d = -f 1)
  if test "$update" ; then echo "Updating $lib" ; pip install -U "$lib" ; else echo "$lib is up-to-date" ; fi
done

# This is a crutch until the globus backend is merged into ONE main
pip install -U git+https://github.com/int-brain-lab/ONE.git@ibl_prod -q

deactivate
