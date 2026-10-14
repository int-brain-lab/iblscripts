#!/bin/bash

# Create or update the mpci env, which runs the mesoscope tasks (see ibllib.pipes.routing)
ENVDIR="$HOME/Documents/PYTHON/envs/mpci"

if [ ! -d "$ENVDIR" ]; then
  echo "$ENVDIR does not exist; creating"
  python3.12 -m venv $ENVDIR
fi

# Optional canary branch files, as in update_iblenv.sh
SUBJECTS=/mnt/s0/Data/Subjects
ibllib_branch="master"
[ -f "$SUBJECTS/.canary_branch" ] && ibllib_branch="$(<$SUBJECTS/.canary_branch)"
one_branch="main"
[ -f "$SUBJECTS/.one_canary_branch" ] && one_branch="$(<$SUBJECTS/.one_canary_branch)"
mpci_branch="main"
[ -f "$SUBJECTS/.mpci_canary_branch" ] && mpci_branch="$(<$SUBJECTS/.mpci_canary_branch)"
printf "\nSetting mpci to branch $mpci_branch, ibllib to branch $ibllib_branch, ONE-api to branch $one_branch\n"

source "$ENVDIR/bin/activate"
pip install --upgrade pip
pip install uv

uv pip uninstall mpci suite2p ;
# Remove any leftover package dir (e.g. stale numba caches), which would make mpci a namespace package
rm -rf "$ENVDIR"/lib/python3*/site-packages/mpci "$ENVDIR"/lib/python3*/site-packages/__editable__*mpci*
uv pip install "mpci[suite2p] @ git+https://github.com/int-brain-lab/mpci.git@$mpci_branch" ;
# mpci depends on ibllib develop; reinstall ONE and ibllib from the same branches as the base env
uv pip uninstall ONE-api ibllib ;
uv pip install "ONE-api @ git+https://github.com/int-brain-lab/ONE.git@$one_branch" ;
uv pip install "ibllib @ git+https://github.com/int-brain-lab/ibllib.git@$ibllib_branch" ;
uv pip install "project_extraction[passiveVideo] @ git+https://github.com/int-brain-lab/project_extraction.git" ;
deactivate
