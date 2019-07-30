#!/usr/bin/env bash
# This is an example of how to run the sync protocol
# run `pip install opencv-python` to install cv2 dependency on top of ibl environment
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/ephys/
source ~/Documents/PYTHON/envs/iblenv/bin/activate
python synchronization_protocol.py /datadisk/Local/20190710_sync_test
