#!/bin/bash
source ~/Documents/PYTHON/envs/iblenv/bin/activate
python ~/Documents/PYTHON/iblscripts/analysis_crons/s3_logs/process_logs.py $1 $2
