FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y software-properties-common \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git wget sudo ninja-build \
        python3-opencv python3-dev python3-pip \
        unzip zip \
        vim \
    && ln -sv /usr/bin/python3 /usr/bin/python

RUN apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb

RUN pip3 install torch==1.10.2+cu111 torchvision==0.11.3+cu111 torchaudio==0.10.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip3 install matplotlib scipy Pillow numpy==1.18.5 prettytable easydict scikit-learn pyyaml yacs termcolor tabulate tensorboard opencv-python pyyaml yacs termcolor \
    scikit-learn tabulate gdown faiss-gpu cython

CMD ["/bin/bash"]