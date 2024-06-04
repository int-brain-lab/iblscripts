#!/bin/bash
set -e

# $1: commit or branch to test out for ibllib
# $2: path to repository, e.g. ~/Documents/github/ibllib-repo
# $3: path to the log directory, e.g. ~/.ci/reports/5cbfa640...

# Update ibllib in github folder in order to flake
pushd $2
git fetch --all
git reset --hard HEAD
git checkout $1
branch=$(git name-rev --name-only $1)

#echo $branch

# Update iblscripts
cd ../iblscripts
git fetch --all
git reset --hard HEAD
# Attempt to checkout same branch name as ibllib; fallback to dev
# if ibllib commit is on master or branch doesn't exist in iblscripts...
if [[ "$branch" =~ ^(remotes\/origin\/)?master$ ]] || \
   [[ "$branch" =~ ^remotes\/origin\/HEAD$ ]] || \
   ! git rev-parse -q --verify --end-of-options $branch; then
        echo Checking out develop branch of iblscripts
        git checkout develop
else
        echo Checking out $branch of iblscripts
        git checkout $branch
fi
# If not detached, pull latest
if git symbolic-ref -q HEAD; then
  git pull --strategy-option=theirs
fi
popd

# second step is to re-install these into the environment
#source ~/Documents/PYTHON/envs/iblenv-ci/bin/activate
source /home/experiment/anaconda3/etc/profile.d/conda.sh
#conda update conda --yes --quiet
conda remove --name ci --all --yes
conda create -n ci --yes --quiet python=3.10

conda activate ci

pip install -U phylib
pip install -U ONE-api
pip install "ibllib[wfield] @ git+https://github.com/int-brain-lab/ibllib.git@$1"
pip install -e /home/experiment/Documents/github/iblscripts
pip install pyfftw
pip install -U git+https://github.com/int-brain-lab/project_extraction.git

# install our root certificate into the python certifi keychain in order to
# recorgnise the signed local alyx SSL certificate
chainfile=$(python -c "import certifi; print(certifi.where())")
printf "\n# Issuer: CN=IBL\n" >> $chainfile
cat /var/www/alyx-test/certs/testCA.pem >> $chainfile

# download the integration data
python /home/experiment/Documents/github/iblscripts/ci/download_data.py
