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
outdated=$(pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1)

# These libraries need to be installed in order, so if one is updated, the ones after need to be updated too
update=$(echo $outdated | grep -o "tensorflow " | cut -d = -f 1)
# Note that the space after tensorflow is crucial as the tensorflow-estimator otherwise keeps tensorflow being updated
if test "$update" ; then
  echo "Updating tensorflow, deeplabcut and ibllib" ;
  pip uninstall -y ONE-api ibllib deeplabcut tensorflow ;
  pip install tensorflow ;
  pip install deeplabcut ;
  pip install ibllib ;
  pip install ONE-api ;
else
  echo "tensorflow is up-to-date" ;
  update=$(echo $outdated | grep -o "deeplabcut" | cut -d = -f 1)
  if test "$update" ; then
    echo "Updating deeplabcut and ibllib" ;
  pip uninstall -y ONE-api ibllib deeplabcut ;
  pip install deeplabcut ;
  pip install ibllib ;
  pip install ONE-api ;
  else
    echo "deeplabcut is up-to-date" ;
    update=$(echo $outdated | grep -o "ibllib" | cut -d = -f 1)
    if test "$update" ; then
      echo "Updating ibllib" ;
      pip uninstall -y ONE-api ibllib ;
      pip install ibllib ;
      pip install ONE-api ;
    else
      echo "ibllib is up-to-date" ;
      update=$(echo $outdated | grep -o "ONE-api" | cut -d = -f 1)
      if test "$update" ; then
        echo "Updating ONE-api" ;
        pip uninstall -y ONE-api ;
        pip install ONE-api ;
      else
        echo "ONE-api is up-to-date" ;
      fi
    fi
  fi
fi

deactivate
