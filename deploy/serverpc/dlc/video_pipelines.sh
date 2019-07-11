#!/usr/bin/env bash
export PATH=/usr/local/cuda-9.0/bin:$PATH;
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# python video_pipelines.py create_flags /mnt/s0/Data/Subjects/ --dry=False
python video_pipelines.py dlc_training /mnt/s0/Data/Subjects/ --dry=True --count=1
