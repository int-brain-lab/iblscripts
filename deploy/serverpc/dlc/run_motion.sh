#!/bin/bash
set -e

source $1
python `dirname "$0"`/run_motion.py $2 $3