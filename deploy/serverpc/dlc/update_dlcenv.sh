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
fi

# Check if any pip libraries are out of date if yes, update
source ~/Documents/PYTHON/envs/dlcenv/bin/activate

outdated=$(pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1)
for lib in "ibllib" "tensorflow" "deeplabcut"
do
  update=$(echo $outdated | grep $lib | cut -d = -f 1)
  if test "$update" ; then pip install -U $lib ; else echo "$lib is up-to-date" ; fi
done

deactivate