#!/bin/bash
set -e
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/crontab
./update_ibllib.sh


./01_extract_training.sh; ./02_register.sh
./20_extract_ephys.sh; ./02_register.sh
./03_compress_videos.sh; ./02_register.sh &
./04_audio_training.sh; ./02_register.sh &
./21_raw_ephys_qc.sh; ./23_compress_ephys.sh; ./27_compress_ephys_videos.sh; ./02_register.sh &
