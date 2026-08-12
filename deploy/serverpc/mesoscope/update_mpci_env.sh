#!/bin/bash

# Make sure local suite2p repository is up to date
ENVDIR="$HOME/Documents/PYTHON/envs/mpci"

if [ ! -d "$ENVDIR" ]; then
  echo "$ENVDIR does not exist; creating"
  python3.12 -m venv $ENVDIR
fi
ENVDIR="$ENVDIR/bin/activate"  # NB: can't guarantee this path will be correct

source "$ENVDIR"
pip install --upgrade pip
pip install uv

uv pip uninstall mpci ;
uv pip install "mpci[suite2p] @ git+https://github.com/int-brain-lab/mpci.git" ;
uv pip uninstall ibllib ;
uv pip install git+https://github.com/int-brain-lab/ibllib.git@mpciPackage ;
uv pip install "project_extraction[passiveVideo] @ git+https://github.com/int-brain-lab/project_extraction.git" ;
deactivate
