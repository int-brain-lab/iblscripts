# auto-update pykilosort
cd ~/Documents/PYTHON/SPIKE_SORTING/pykilosort
git fetch --all
git checkout -f ibl_prod
git reset --hard
git pull

# auto-update the environment
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate pyks2
pip uninstall -y ibllib
pip install git+https://github.com/int-brain-lab/ibllib.git@$pykilosort
pip install -U phylib
