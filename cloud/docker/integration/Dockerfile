#docker build -t ibl/iblenv:base
#Run above to build the image yourself.

FROM continuumio/miniconda3
LABEL maintainer="International Brain Lab"
LABEL version="0.1"
LABEL description="Parent image for containers using iblenv, inherits miniconda and ubuntu"

WORKDIR /data

RUN git clone https://github.com/int-brain-lab/iblenv /data/iblenv
RUN apt install libgl1-mesa-glx -y
RUN git clone https://github.com/int-brain-lab/ibllib.git
RUN git clone https://github.com/cortex-lab/phy.git
RUN git clone https://github.com/cortex-lab/phylib.git

RUN conda install -y mamba -c conda-forge
RUN mamba env update --name base --file ./iblenv/iblenv.yaml
RUN mamba install conda-build
RUN mamba clean --all
RUN conda develop ./ibllib && conda develop ./phylib && conda develop ./phy
RUN cd ibllib && git checkout develop
