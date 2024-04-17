#!/bin/bash

# Check if there are canary_branch files for ibllib or one
# canary_branch is an optional text file containing the ibllib/iblscripts branch to be installed
FILE=/mnt/s0/Data/Subjects/.canary_branch
if [ -f "$FILE" ]; then
    ibllib_branch="$(<$FILE)"
    printf "\n$FILE exists. Setting ibllib to branch $ibllib_branch\n"
else
    ibllib_branch="master"
fi

# one_canary_branch is an optional text file containing the one branch to be installed
FILE=/mnt/s0/Data/Subjects/.one_canary_branch
if [ -f "$FILE" ]; then
    one_branch="$(<$FILE)"
    printf "\n$FILE exists. Setting ONE-api to branch $one_branch\n"
else
    one_branch="main"
fi

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
outdated=$(pip list --outdated | awk 'NR>2 {print $1}')

# Check if pip needs update
update=$(echo $outdated | grep -o "pip" | cut -d = -f 1)
if test "$update" ; then
  echo "Updating pip" ;
  pip install --upgrade pip
fi

# Note: no updating for tensorflow and deeplabcut for now because this has been messing with our env
## These libraries need to be installed in order, so if one is updated, the ones after need to be updated too
#update=$(echo $outdated | grep -o "tensorflow " | cut -d = -f 1)
## Note that the space after tensorflow is crucial as the tensorflow-estimator otherwise keeps tensorflow being updated
#if test "$update" ; then
#  echo "Updating tensorflow, deeplabcut and ibllib" ;
#  pip uninstall -y ONE-api ibllib deeplabcut tensorflow ;
#  pip install tensorflow ;
#  pip install deeplabcut ;
#  pip install ibllib ;
#  pip install ONE-api ;
#else
#  echo "tensorflow is up-to-date" ;
#  update=$(echo $outdated | grep -o "deeplabcut" | cut -d = -f 1)
#  if test "$update" ; then
#    echo "Updating deeplabcut and ibllib" ;
#    pip uninstall -y ONE-api ibllib deeplabcut ;
#    pip install deeplabcut ;
#    pip install ibllib ;
#    pip install ONE-api ;
#  else
#    echo "deeplabcut is up-to-date" ;
update=$(echo $outdated | grep -o "ibllib" | cut -d = -f 1)
if test "$update" || [[ $ibllib_branch != "master" ]] ; then
  echo "Updating ibllib" ;
  pip uninstall -y ONE-api ibllib ;
  pip install git+https://github.com/int-brain-lab/ibllib.git@$ibllib_branch ;
  pip install git+https://github.com/int-brain-lab/ONE.git@$one_branch ;
else
  echo "ibllib is up-to-date" ;
  update=$(echo $outdated | grep -o "ONE-api" | cut -d = -f 1)
  if test "$update" || [[ $one_branch != "main" ]] ; then
    echo "Updating ONE-api" ;
    pip uninstall -y ONE-api ;
    pip install git+https://github.com/int-brain-lab/ONE.git@$one_branch ;
  else
    echo "ONE-api is up-to-date" ;
  fi
fi
#  fi
#fi
# Update ibl libraries
update=$(echo $outdated | grep -o "iblutil" | cut -d = -f 1)
if test "$update" ; then
  pip install --upgrade iblutil
fi
update=$(echo $outdated | grep -o "ibl-neuropixel" | cut -d = -f 1)
if test "$update" ; then
  pip install --upgrade ibl_neuropixel
fi
update=$(echo $outdated | grep -o "iblatlas" | cut -d = -f 1)
if test "$update" ; then
  pip install --upgrade iblatlas
fi

deactivate
