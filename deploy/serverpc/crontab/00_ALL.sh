#!/bin/bash
set -e
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
./01_extract.sh
./02_register.sh
./03_compress.sh
./02_register.sh
