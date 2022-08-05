#!/bin/bash
# Similar to 01_run_coverage.sh but pipes lcov output to Node coveralls
set -e

# ./02_run_coverage.sh master /home/experiment/Documents/github/ibllib-repo /home/experiment/.ci
# $1: commit or branch to test out for ibllib
# $2: path to ibllib - assumes matlab-ci and iblscripts are in the same parent directory
# $3: path to output log directory

source /home/experiment/anaconda3/etc/profile.d/conda.sh
conda activate ci

# Flake ibllib and save the output in a separate log
mkdir -p "$3"
pushd "$2"
flake8 . --tee --output-file="$3/flake_output.txt"
popd

coverage erase  # Be extra careful and erase previous coverage
# Should run the tests in the source directory as the coverage paths are relative to the cwd.
# As we install via pip this will be in anaconda3 env packages
#
# NB: If we ever need to change to a different directory we can call renameHTML and pass the
# filepath arg to node-coveralls, specifying the location relative to our working directory.
source=$(pip show ibllib | awk -F':' '$1 == "Location" { print $2 }' | xargs)
cd "$source"

# Build up sources
pkgs=''
for pkg in 'ibllib' 'brainbox'
do
   #pkgs+="${source}/${pkg},"
   pkgs+="${pkg},"
done
pkgs=${pkgs%?} # remove last comma

# Run tests
coverage run --source="$pkgs" --rcfile "$2/../iblscripts/.coveragerc" \
"$2/../iblscripts/runAllTests.py" -c "$1" -r "$2" --logdir "$3"

#coverage report --skip-covered # (for debugging)

echo Saving coverage reports
coverage html -d "$3/reports/$1" --skip-covered --show-contexts
coverage xml -o "$3/reports/$1/CoverageResults.xml"
coverage json -o "$3/reports/$1/CoverageResults.json"
coverage lcov -o "$3/reports/$1/CoverageResults.lcov"

# Post lcov to coveralls then delete file
coveralls=$("$2/../matlab-ci/node_modules/coveralls/bin/coveralls.js")
coveralls -v < "$3/reports/$1/CoverageResults.lcov" && rm "$3/reports/$1/CoverageResults.lcov"
