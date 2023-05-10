#!/bin/bash
set -e
# Check if there are canary_branch files for ibllib or one
# canary_branch is an optional text file containing the ibllib/iblscripts branch to be installed
FILE=/mnt/s0/Data/Subjects/.canary_branch
if [ -f "$FILE" ]; then
    ibllib_branch="$(<$FILE)"
    printf "\n$FILE exists. Setting ibllib to branch $ibllib_branch\n"
else
    ibllib_branch="master"
fi

# Check if iblscripts is up to date
cd ~/Documents/PYTHON/iblscripts
git fetch --all -p
# Attempt to checkout same branch name as ibllib; fallback to master
# if ibllib commit is on master or branch doesn't exist in iblscripts...
if [[ "$ibllib_branch" =~ ^(remotes\/origin\/)?master$ ]] || \
   [[ "$ibllib_branch" =~ ^remotes\/origin\/HEAD$ ]] || \
   ! git rev-parse -q --verify --end-of-options $ibllib_branch; then
        echo "Checking out master branch of iblscripts"
        git checkout -f master
else
        echo "Checking out $ibllib_branch of iblscripts"
        git checkout -f $ibllib_branch
fi
git reset --hard -q
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "@{u}")
if [ "$LOCAL" != "$REMOTE" ]; then
  echo "Updating iblscripts"
  git pull
else
  echo "iblscripts is up-to-date"
fi

# one_canary_branch is an optional text file containing the one branch to be installed
FILE=/mnt/s0/Data/Subjects/.one_canary_branch
if [ -f "$FILE" ]; then
    one_branch="$(<$FILE)"
    printf "\n$FILE exists. Setting ONE-api to branch $one_branch\n"
else
    one_branch="main"
fi

# Make sure we are in ibllib env
source ~/Documents/PYTHON/envs/iblenv/bin/activate

# Collect outdated pip packages
outdated=$(pip list --outdated | awk 'NR>2 {print $1}')

# Check if pip or phylib needs update
for lib in "pip" "phylib"
do
  update=$(echo $outdated | grep -o $lib | cut -d = -f 1)
  if test "$update" ; then
    printf "\nUpdating $lib" ;
    pip install --upgrade $lib ;
  else
    printf "\n$lib is up-to-date" ;
  fi
done

# Uninstall and clean reinstall ibl libraries. This is necessary in case of canary branches and to keep servers
# up to date with latest versions
printf "\nUninstalling and reinstalling ONE-api, ibllib and project_extraction.\n"
pip uninstall -y ONE-api ibllib project_extraction ;
pip install git+https://github.com/int-brain-lab/ONE.git@$one_branch ;
pip install git+https://github.com/int-brain-lab/ibllib.git@$ibllib_branch --upgrade-strategy eager ;
pip install git+https://github.com/int-brain-lab/project_extraction.git ;
