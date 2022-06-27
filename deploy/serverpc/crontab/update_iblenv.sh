#!/bin/bash
set -e
# Check if iblscripts is up to date
cd ~/Documents/PYTHON/iblscripts
git checkout -f master -q
git reset --hard -q
git fetch
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "@{u}")
if [ $LOCAL != $REMOTE ]; then
  echo "Updating iblscripts"
  git pull
else
  echo "iblscripts is up-to-date"
fi


# Check if there are canary_branch files for ibllib or one
# canary_branch is an optional text file containing the ibllib branch to be installed
FILE=/mnt/s0/Data/Subjects/.canary_branch
if [ -f "$FILE" ]; then
    ibllib_branch="$(<$FILE)"
    echo "$FILE exists. Setting ibllib to branch $ibllib_branch"
else
    ibllib_branch="master"
fi
# one_canary_branch is an optional text file containing the one branch to be installed
FILE=/mnt/s0/Data/Subjects/.one_canary_branch
if [ -f "$FILE" ]; then
    one_branch="$(<$FILE)"
    echo "$FILE exists. Setting ONE-api to branch $one_branch"
else
    one_branch="ibl_prod"
fi

# Make sure we are in ibllib env
source ~/Documents/PYTHON/envs/iblenv/bin/activate
# Check if simple installed pip libraries are out of date, if yes, update
outdated=$(pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1)
# Check if pip needs update
update=$(echo $outdated | grep -o "pip" | cut -d = -f 1)
if test "$update" ; then
  echo "Updating pip" ;
  pip install --upgrade pip
fi

# Check if phylib needs update, if yes, update all
update=$(echo $outdated | grep -o "phylib" | cut -d = -f 1)
if test "$update" ; then
  echo "Updating phylib, ONE and ibllib" ;
  pip uninstall -y phylib ONE-api ibllib;
  pip install phylib ;
  pip install git+https://github.com/int-brain-lab/ONE.git@$one_branch ;
  pip install git+https://github.com/int-brain-lab/ibllib.git@$ibllib_branch --upgrade-strategy eager
else
  echo "phylib is up-to-date" ;
  #update=$(echo $outdated | grep -o "deeplabcut" | cut -d = -f 1)
fi


pip install phylib
pip install git+https://github.com/int-brain-lab/ONE.git@$one_branch
pip install git+https://github.com/int-brain-lab/ibllib.git@$ibllib_branch --upgrade-strategy eager
pip install -U git+https://github.com/int-brain-lab/project_extraction.git
