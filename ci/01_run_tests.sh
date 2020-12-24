
#!/bin/bash
set -e

# ./01_run_tests.sh develop  /home/ibladmin/Documents/CI/ibllib-repo /home/ibladmin/Documents/CI
# $1: commit or branch to test out for ibllib
# $2: path to ibllib
# $3: path to output log directory

source ~/Documents/PYTHON/envs/iblenv-ci/bin/activate
python /home/ibladmin/Documents/CI/iblscripts/runAllTests.py -c "$1" -r "$2" --logdir "$3"
