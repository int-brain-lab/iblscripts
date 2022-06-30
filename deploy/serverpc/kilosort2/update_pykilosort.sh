#!/bin/bash

# Make sure local pykilosort repository is up to date
cd ~/Documents/PYTHON/SPIKE_SORTING/pykilosort
git checkout -f ibl_prod -q
git reset --hard -q
git fetch
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "@{u}")
if [ $LOCAL != $REMOTE ]; then
  echo "Updating pykilosort"
  git pull
else
  echo "pykilosort is up-to-date"
fi

# Check that all libraries in the env are up to date
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate pyks2
outdated=$(pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1)

# Check if pip needs update
update=$(echo $outdated | grep -o "pip" | cut -d = -f 1)
if test "$update" ; then
  echo "Updating pip" ;
  pip install --upgrade pip
fi

# Libraries that have to be updated in order
update=$(echo $outdated | grep -o "phylib" | cut -d = -f 1)
if test "$update" ; then
  echo "Updating phylib and ibllib" ;
  pip uninstall -y ibllib phylib ;
  pip install phylib ;
  pip install ibllib ;
else
echo "phylib is up-to-date" ;
fi

update=$(echo $outdated | grep -o "ibllib" | cut -d = -f 1)
if test "$update" ; then
  echo "Updating ibllib" ;
  pip uninstall -y ibllib ;
  pip install ibllib ;
else
  echo "ibllib is up-to-date" ;
fi

update=$(echo $outdated | grep -o "ibl-neuropixel" | cut -d = -f 1)
if test "$update" ; then
  echo "Updating ibl-neuropixel" ;
  pip uninstall -y ibl-neuropixel ;
  pip install ibl-neuropixel ;
else
  echo "ibl-neuropixel is up-to-date" ;
fi

conda deactivate
