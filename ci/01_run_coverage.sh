#!/bin/bash
#set -e

# ./02_run_coverage.sh master /home/experiment/Documents/github/ibllib-repo /home/experiment/.ci
# $1: commit or branch to test out for ibllib
# $2: path to ibllib
# $3: path to output log directory

source /home/experiment/anaconda3/etc/profile.d/conda.sh
conda activate ci

# Flake ibllib and save the output in a separate log
mkdir -p $3
pushd $2
flake8 . --tee --output-file="$3/flake_output.txt"
popd

coverage erase  # Be extra careful and erase previous coverage
# Running tests
source=$(pip show ibllib | awk -F':' '$1 == "Location" { print $2 }' | xargs)

pkgs=''
for pkg in 'ibllib' 'oneibl' 'brainbox' 'alf'
do
   #pkgs+="${source}/${pkg},"
   pkgs+="${pkg},"
done
pkgs=${pkgs%?} # remove last comma

# Build up sources
coverage run --omit=*pydevd_file_utils.py,*test_* --source="$pkgs" \
/home/experiment/Documents/github/iblscripts/runAllTests.py -c "$1" -r "$2" --logdir "$3"
#coverage report --skip-covered

echo Saving coverage reports
coverage html -d "$3/reports/$1" --skip-covered
coverage xml -o "$3/reports/$1/CoverageResults.xml"

# Remove source directories from HTML report (more readable/secure)
echo Renaming coverage files
python /home/experiment/Documents/github/iblscripts/ci/renameHTML.py -d "$3/reports/$1" -s "$source"
