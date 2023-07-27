#!/bin/bash
set -e

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64:$LD_LIBRARY_PATH

source $1
python `dirname "$0"`/run_litpose.py $2 $3