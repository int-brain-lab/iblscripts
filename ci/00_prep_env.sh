#!/bin/bash
set -e

# $1: commit or branch to test out for ibllib
# $2: path to repository, e.g. ~/Documents/github/ibllib-repo (assumes iblscripts in same parent dir)
# $3: path to the log directory, e.g. ~/.ci/reports/5cbfa640...
#
# NB: The following paths are hardcoded and specific to the cortexlab ci:
# 1. $HOME/anaconda3/etc/profile.d/conda.sh
# 2. $HOME/Documents/github/iblscripts
# 3. /var/www/alyx-test/certs/testCA.pem
# 4. /var/www/alyx-test/certs/certlink.sh
# 5. $HOME/Documents/github/iblscripts/ci/download_data.py

# Update ibllib in github folder in order to flake
pushd "$2"  # into ibllib folder
git fetch --all
git reset --hard HEAD
git checkout $1
branch=$(git name-rev --name-only $1)

#echo $branch

# Update iblscripts
iblscripts="$(dirname "$2")/iblscripts"
cd "$iblscripts"
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
source "$HOME/anaconda3/etc/profile.d/conda.sh"
#conda update conda --yes --quiet
conda remove --name ci --all --yes
conda create -n ci --yes --quiet python=3.10

conda activate ci

pip install -U phylib
pip install -U ONE-api
pip install "ibllib[wfield] @ git+https://github.com/int-brain-lab/ibllib.git@$1"
pip install -e "$iblscripts"
pip install pyfftw
pip install -U git+https://github.com/int-brain-lab/project_extraction.git

# install our root certificate into the python certifi keychain in order for
# request package to recognise the signed local alyx SSL certificate
chainfile=$(python -c "import certifi; print(certifi.where())")
certfile=/var/www/alyx-test/certs/testCA.pem
printf "\n# Issuer: CN=IBL\n" >> "$chainfile"
cat "$certfile" >> "$chainfile"
# install to env version of openssl (may be different to system one) so that
# urllib recognises the SSL cert
# copy to other certs (used by openssl and therefore urllib python package)
# get location of openssl directory containing certs folder
cadir=$(openssl version -d | sed -n '/ *OPENSSLDIR *: *"/ { s///; s/".*//; p; q;}')
if [ -d "$cadir/certs" ]; then
  # certs folder exists with a load of certs symlinks
  echo "Linking CA cert to $cadir/certs"
  ln -s "$certfile" "$cadir/certs/testCA.pem"
  # Make hash link just in case it's required
  /bin/bash /var/www/alyx-test/certs/certlink.sh "$cadir/certs/testCA.pem"
elif [ -e "$cadir/cert.pem" ] && ! grep -Fxq "Issuer: CN=IBL" "$cadir/cert.pem"; then
  # A cert.pem file exists containing all certs
  echo "Appending to $cadir/cert.pem"
  printf "\nIssuer: CN=IBL\n==============\n" >> "$cadir/cert.pem"
  cat "$certfile" >> "$cadir/cert.pem"
fi

# download the integration data
python "$iblscripts/ci/download_data.py"
