#!/bin/bash
set -e

# ./01_run_tests.sh develop  /home/ibladmin/Documents/CI/ibllib-repo /home/ibladmin/Documents/CI
# $1: commit or branch to test out for ibllib
# $2: path to ibllib
# $3: path to output log directory

#source ~/Documents/PYTHON/envs/iblenv-ci/bin/activate
source /home/experiment/anaconda3/etc/profile.d/conda.sh
conda activate ci

# Flake ibllib and save the output in a separate log
mkdir -p $3
pushd $2
flake8 . --tee --output-file="$3/flake_output.txt"
popd

python /home/experiment/Documents/github/iblscripts/runAllTests.py -c "$1" -r "$2" --logdir "$3"
