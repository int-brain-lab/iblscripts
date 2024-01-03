#!/bin/bash

### Pull the docker container
sudo docker pull apasarkar/localmd

### args from MesoscopePMDCompress task
# host refers to paths on whiterussian
host_data_path=$1
host_ops_path=$2
host_out_path=$3
block_height=$4
block_width=$5

# these are locations used within the docker filesystem
dataset_target="/data.bin"
script_target="/script.py"
ops_target="/ops.npy"
output_target="/output/"

echo $host_data_path
echo $host_ops_path
echo $host_out_path

### Launch the container
# the bin and ops files, the run_localmd.py Python script, and the output directory are mounted 
# then the localmd.py script is run via bash on the container
docker run --gpus=all \
            --mount type=bind,source="/home/ibladmin/Documents/PYTHON/iblscripts/deploy/mesoscope/run_localmd.py",destination=$script_target \
            --mount type=bind,source=$host_data_path,destination=$dataset_target \
            --mount type=bind,source=$host_ops_path,destination=$ops_target \
            --mount type=bind,source=$host_out_path,destination=$output_target \
            -v $1:/app/ \
            apasarkar/localmd -c "set -eo pipefail && source /venv/bin/activate &&mkdir /apps/ && python script.py --dataset_path $dataset_target --ops_file $ops_target --block_height $block_height --block_width $block_width --output_location $output_target"

