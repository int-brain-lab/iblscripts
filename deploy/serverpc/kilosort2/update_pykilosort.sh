#!/bin/bash

# Make sure local pykilosort repository is up to date
cd ~/Documents/PYTHON/SPIKE_SORTING/pykilosort
git checkout -f ibl_prod -q
git reset --hard origin/ibl_prod -q
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "@{u}")
if [ $LOCAL != $REMOTE ]; then
  print("Updating pykilosort")
  git pull
fi

# Check that all libraries in the env are up to date
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate pyks2

outdated=$(pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1)
for lib in "ibllib" "phylib"
do
  update=$(echo $outdated | grep $lib | cut -d = -f 1)
  if test "$update" ; then pip install -U $lib ; else echo "$lib is up-to-date" ; fi
done
conda deactivate