#!/bin/bash

# Check if any pip libraries are out of date if yes, update
source ~/Documents/PYTHON/envs/dlcenv/bin/activate
echo "Updating ibllib mesoscope"
pip uninstall -y ibllib
pip install git+https://github.com/int-brain-lab/ibllib.git@mesoscope
