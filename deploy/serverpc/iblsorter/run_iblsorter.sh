#!/bin/bash
set -e

SCRATCH_DRIVE=/home/olivier/scratch  # this is the path of the scratch SSD volume for intermediate KS2 results swapping
# --------------- DO NOT EDIT BELOW

source ~/Documents/PYTHON/SPIKE_SORTING/ibl-sorter/.venv/bin/activate
which python  # checks the version for easier debugging
python ~/Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2/run_pykilosort.py $1 $2
