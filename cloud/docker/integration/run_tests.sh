# docker run -v /mnt/s0/Data/IntegrationTests:/mnt/s0/Data/IntegrationTests -it integration_ibllib
#TODO put this into the Dockerfile and recompile
apt-get update
apt-get install -y python3-tk

cd home

#TODO remove develop branch for iblscript
git clone --branch develop https://github.com/int-brain-lab/iblscripts.git
pip install git+https://github.com/cortex-lab/phylib.git@ibl_tests
pip install git+https://github.com/int-brain-lab/ibllib.git@integration
cp /mnt/s0/Data/IntegrationTests/.one_params /root/.one_params

pytest ./iblscripts/tests
#TODO gather tests results and setup a webhook
