#!/bin/bash
set -e
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
./update_ibllib.sh


./01_extract_training.sh; ./02_register.sh
./20_extract_ephys.sh; ./02_register.sh
./03_compress_videos.sh; ./02_register.sh
./04_audio_training.sh; ./02_register.sh
