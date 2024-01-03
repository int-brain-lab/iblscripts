#!/bin/bash

#Chris: The below script assumes you have a bin file and you have an ops file. I have modified the IBL script to also reflect that. I think that this run script should do all the work to get the data into that format; so for e.g. if there is no ops file, you can insert logic here to detect that and accordingly add a dummy ops file with Ly Lx and nframes as the fields. 

#Here are the inputs to this script: 

# $1 save_dir (the location you want to save the results to)
# $2 bin_file_path (the location of the actual dataset ON HOST) 
# $3 ops_file_path (the location of the actual ops_file ON HOST)
# $4 block height (user specified). int
# $5 block width (user specified). int

#In the run_localmd_kickoff, I show concretely how you can run run this script, and it is super easy to modify and tailor to the IBL workflow needs 

#One detail: it is entirely possible that you will have to activate 
#'''
###
### Logic pertaining to experimental details (FOV, some logic to create a 'dummy' ops file containing ops['Lx'], ops['Ly'], and ops['nframes'] if ops file doesn't exist
### should go here
### Below the execution script uses entirely generic names and this run_localmd.sh script should manage the naming/other data wrangling. 
###

##New Params


### Pull the docker container if you have not already done so. This step gets skipped if your system already has it (avoiding time consuming download every time)
sudo docker pull apasarkar/localmd

### Specify a folder or location on your HOST system where this data should be saved: 
host_data_path=$1
host_ops_path=$2
host_out_path=$3
block_height=$4
block_width=$5

#These are locations used within the docker filesystem
dataset_target="/data.bin"
script_target="/script.py"
ops_target="/ops.npy"
output_target="/output/"

echo $host_data_path
echo $host_ops_path
echo $host_out_path

### Launch the container
docker run --gpus=all \
            --mount type=bind,source="/home/ibladmin/Documents/PMDTEST/iblscripts/deploy/mesoscope/RUN_LOCALMD.py",destination=$script_target \
            --mount type=bind,source=$host_data_path,destination=$dataset_target \
            --mount type=bind,source=$host_ops_path,destination=$ops_target \
            --mount type=bind,source=$host_out_path,destination=$output_target \
            -v $1:/app/ \
            apasarkar/localmd -c "set -eo pipefail && source /venv/bin/activate &&mkdir /apps/ && python script.py --dataset_path $dataset_target --ops_file $ops_target --block_height $block_height --block_width $block_width --output_location $output_target"

#docker run --gpus=all \
#           --mount type=bind,source="/home/ibladmin/Documents/PMDTEST/iblscripts/deploy/mesoscope/RUN_LOCALMD.py",destination=$script_target \
#           --mount type=bind,source=$host_data_path,destination=$dataset_target \
#           --mount type=bind,source="/data/home/app2139/localmd/IBL_Development/entrypoint_IBL.sh",destination="/entrypoint_IBL.sh" \
#           --mount type=bind,source=$host_ops_path,destination=$ops_target \
#           -v $1:/apps/ \
#           --entrypoint /entrypoint_IBL.sh apasarkar/localmd 

#docker run --gpus=all \
#--mount type=bind,source="/home/ibladmin/Documents/PMDTEST/iblscripts/deploy/mesoscope/RUN_LOCALMD.py",destination=$script_target
