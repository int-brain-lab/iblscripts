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
echo "flaking ibllib"
flake8 . --tee --output-file="$3/reports/$1/flake_output_ibllib.txt"

# Flake iblscripts
echo "flaking iblscripts"
flake8 "$2/../iblscripts" --tee --output-file="$3/reports/$1/flake_output_iblscripts.txt" --config="$2/../iblscripts/.flake8"

# Merge flake reports
for file in "$3"/reports/"$1"/flake_output_*.txt
do
   printf '===== %s =====\n\n' "$file" >> "$3/reports/$1/flake_output.txt"
   cat "$file" >> "$3/reports/$1/flake_output.txt"
   rm "$file"
done

# Should run the tests in the source directory as the coverage paths are relative to the cwd.
# As we install via pip this will be in anaconda3 env packages
#
# NB: If we ever need to change to a different directory we can call renameHTML and pass the
# filepath arg to node-coveralls, specifying the location relative to our working directory.
source=$(pip show ibllib | awk -F':' '$1 == "Location" { print $2 }' | xargs)
pushd "$source"

coverage erase  # Be extra careful and erase previous coverage

# Build up sources
pkgs=''
for pkg in 'ibllib' 'brainbox'
do
   #pkgs+="${source}/${pkg},"
   pkgs+="${pkg},"
done
pkgs=${pkgs%?} # remove last comma

# Run tests
passed=true
coverage run --source="$pkgs" --rcfile "$2/../iblscripts/.coveragerc" \
"$2/../iblscripts/runAllTests.py" -c "$1" -r "$2" --logdir "$3" --exit || passed=false

#coverage report --skip-covered # (for debugging)

echo Saving coverage reports
coverage html -d "$3/reports/$1" --skip-covered --show-contexts
coverage xml -o "$3/reports/$1/CoverageResults.xml"
coverage json -o "$3/reports/$1/CoverageResults.json"
if ! [ "${passed}" = false ] ; then
   echo "Done"
   exit
fi

# Post lcov to coveralls then delete file
coverage lcov -o "$3/reports/$1/CoverageResults.lcov"
popd > /dev/null

echo Posting to coveralls.io
coveralls="$2/../matlab-ci/node_modules/coveralls/bin/coveralls.js"
$coveralls < "$3/reports/$1/CoverageResults.lcov" && rm "$3/reports/$1/CoverageResults.lcov"
popd > /dev/null
echo "Done"
