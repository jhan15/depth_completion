# Copyright 2020 Toyota Research Institute.  All rights reserved.

FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

ENV PROJECT=packnet-sfm
ENV PYTORCH_VERSION=1.8.1+cu111
ENV TORCHVISION_VERSION==0.9.1+cu111
ENV CUDNN_VERSION=8.3.2.*-1+cuda11.5
ENV NCCL_VERSION=2.8.4-1+cuda11.1
ENV HOROVOD_VERSION=65de4c961d1e5ad2828f2f6c4329072834f27661
ENV TRT_VERSION=6.0.1.5
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG python=3.7
ARG cuda_arch="8.6+PTX"
ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Updated: 2022-05-18
# NVIDIA is working on some changes. Some latest CUDA and Ubuntu versions are already working
# (images such as CUDA 11.6 for Ubuntu 20.04 can be rebuild from their code at Gitlab),
# but others (older CUDA/Ubuntu versions such as CUDA 11.2) may still fail.
# Workaround https://github.com/NVIDIA/nvidia-docker/issues/1632
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    libcudnn8=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION} \
    libnccl-dev=${NCCL_VERSION} \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    unzip \
    bzip2 \
    htop \
    gnuplot \
    ffmpeg

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Install Python and pip
RUN apt update \
    && apt install -y python3-pip \
    && python${PYTHON_VERSION} -m pip install --upgrade --force pip

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

# Install Pydata and other deps
RUN pip install future typing numpy pandas matplotlib jupyter h5py \
    awscli boto3 tqdm termcolor path.py pillow-simd opencv-python-headless \
    mpi4py onnx onnxruntime pycuda yacs cython==0.29.10

# Install PyTorch
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html && ldconfig

# Install apex
ENV TORCH_CUDA_ARCH_LIST=${cuda_arch}
RUN mkdir /workspace
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /workspace/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
ENV PYTHONPATH="/workspace/apex:$PYTHONPATH"
WORKDIR /workspace

# Install horovod (for distributed training)
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION} && \
    ldconfig

# Settings for S3
# RUN aws configure set default.s3.max_concurrent_requests 100 && \
#     aws configure set default.s3.max_queue_size 10000

# Install Minkowski Engine
RUN pip install ninja
RUN apt-get update && apt-get install -y libopenblas-dev
RUN export CXX=g++-7
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git
RUN cd /workspace/MinkowskiEngine && \
    python setup.py install --force_cuda

# Add Tini (cf. https://github.com/jupyter/docker-stacks)
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "-g", "--"]

# Install DGP (dataset utils)
# WORKDIR /workspace
# RUN git clone https://github.com/TRI-ML/dgp.git
# WORKDIR /workspace/dgp
# RUN git checkout 9617e65ad351558636de5586a48db848eab578c6
# ENV PYTHONPATH="/workspace/dgp:$PYTHONPATH"
# WORKDIR /workspace
# RUN git clone https://github.com/TRI-ML/dgp.git new_dgp
# RUN cp new_dgp/dgp/utils/pose.py dgp/dgp/utils/
# RUN rm -r -f new_dgp

# Override DGP wandb with required version
RUN pip install wandb==0.8.21 pyquaternion xarray diskcache tenacity pycocotools

# Expose Port for jupyter (8888)
EXPOSE 8888

# Tensorboard
RUN pip install tensorboard tensorflow
EXPOSE 6006

# Create project workspace dir
RUN mkdir -p /workspace/experiments
RUN mkdir -p /workspace/${PROJECT}
WORKDIR /workspace/${PROJECT}

# Copy project source last (to avoid cache busting)
WORKDIR /workspace/${PROJECT}
COPY . /workspace/${PROJECT}
ENV PYTHONPATH="/workspace/${PROJECT}:$PYTHONPATH"
