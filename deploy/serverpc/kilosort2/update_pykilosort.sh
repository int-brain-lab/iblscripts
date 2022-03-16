#!/bin/bash
# auto-update pykilosort
cd /home/ubuntu/Documents/PYTHON/SPIKE_SORTING/pykilosort || exit 1
git fetch --all
git checkout -f ibl_prod
git reset --hard origin/ibl_prod
git pull

# auto-update the environment
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate pyks2
pip uninstall -y ibllib
pip install git+https://github.com/int-brain-lab/ibllib.git@master
pip install -U phylib
