#!/bin/bash

# Once this script is in the desired directory of a newly created instance, a sample command to run:
# sudo bash server_bootstrap.sh mbox

echo "NOTE: Installation log can be found in the directory the script is called from and named 'server_bootstrap_install.log'"
{
  # check to make sure the script is being run with root privileges
  if [ "$(id -u)" != "0" ]; then
    echo "Script needs to be run with sudo or as root, exiting."
    exit 1
  fi

  # check on arguments passed, at least one is required to pick build env
  if [ -z "$1" ]; then
      echo "Error: No argument supplied, script requires first argument for build env (alyx-prod, alyx-dev, openalyx)"
      exit 1
  fi

  # Set vars
  WORKING_DIR=/home/ubuntu/conf_files

  echo "Creating relevant directories and log files..."
  mkdir -p $WORKING_DIR

  echo "Setting hostname of instance..."
  hostnamectl set-hostname "$1"

  echo "Setting timezone to Europe\Lisbon..."
  timedatectl set-timezone Europe/Lisbon

  echo "Removing needrestart package to more easily automate, update apt package index, install awscli and ansible..."
  apt-get --yes remove needrestart
  apt-get --quiet update
  add-apt-repository --yes --update ppa:ansible/ansible
  apt-get --yes --quiet install \
    ansible \
    awscli

#  echo "Copying files from s3 bucket..." # this is dependant on the correct IAM role being applied to the EC2 instance
#  cd $WORKING_DIR || exit 1
#  aws s3 cp s3://alyx-docker/000-default-conf-"$1" .
#  aws s3 cp s3://alyx-docker/apache-conf-"$1" .
#  aws s3 cp s3://alyx-docker/fullchain.pem-"$1" .
#  aws s3 cp s3://alyx-docker/ip-whitelist-conf .
#  aws s3 cp s3://alyx-docker/privkey.pem-"$1" .
#  aws s3 cp s3://alyx-docker/settings.py-"$1" .
#  aws s3 cp s3://alyx-docker/settings_lab.py-"$1" .
#  aws s3 cp s3://alyx-docker/settings_secret.py-"$1" .

  echo "Performing any remaining package upgrades..."
  apt --quiet --yes upgrade  # needs to be apt and not apt-get

  echo "Instance will now reboot..."
  sleep 10s
} | tee -a server_bootstrap_install.log

reboot