#!/bin/bash

# Make sure local suite2p repository is up to date
ENVDIR="$HOME/Documents/PYTHON/envs/suite2p"

if [ -d "$ENVDIR" ]; then
  echo "$ENVDIR does exist; creating"
  python -m venv $ENVDIR
fi
ENVDIR="$ENVDIR/bin/activate"  # NB: can't guarantee this path will be correct

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
/bin/bash "$parent_path/../crontab/update_iblenv.sh" "$ENVDIR"

source "$ENVDIR"
pip uninstall -y suite2p ;
pip install git+https://github.com/samupicard/suite2p.git ;
deactivate
