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

# These libraries need to be installed in order, so if one is updated, the ones after need to be updated too
update=$(echo $outdated | grep -o "tensorflow " | cut -d = -f 1)
# Note that the space after tensorflow is crucial as the tensorflow-estimator otherwise keeps tensorflow being updated
if test "$update" ; then
  echo "Updating tensorflow, deeplabcut and ibllib" ;
  pip uninstall -y ibllib deeplabcut tensorflow ;
  pip install tensorflow ;
  pip install deeplabcut ;
  pip install ibllib ;
else
  echo "tensorflow is up-to-date" ;
  update=$(echo $outdated | grep -o "deeplabcut" | cut -d = -f 1)
  if test "$update" ; then
    echo "Updating deeplabcut and ibllib" ;
  pip uninstall -y ibllib deeplabcut ;
  pip install deeplabcut ;
  pip install ibllib ;
  else
    echo "deeplabcut is up-to-date" ;
    update=$(echo $outdated | grep -o "ibllib" | cut -d = -f 1)
    if test "$update" ; then
      echo "Updating ibllib" ;
      pip uninstall -y ibllib ;
      pip install ibllib ;
    else echo "ibllib is up-to-date" ;
    fi
  fi
fi


# This is a crutch until the globus backend is merged into ONE main
echo "Updating ONE ibl_prod"
pip install -U git+https://github.com/int-brain-lab/ONE.git@ibl_prod -q

deactivate
