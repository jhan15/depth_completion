# Setup

Here describes the setup of the depth completion framework in an EC2 instance.

## System setup

#### Requirements

* ubuntu __18.04__
* cuda __11.1__
* gcc __7.5.0__
* python __3.7__
* docker __20.10__

#### Setup

```bash
# Install nvidia driver (if needed)
$ sudo apt update
$ sudo apt upgrade
$ apt search nvidia-driver
$ sudo apt install nvidia-driver-xxx # that supports cuda 11.1
$ sudo reboot
# Check Driver + GPU
$ nvidia-smi
# Install cuda 11.1 (if needed)
$ wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
$ sudo sh cuda_11.1.1_455.32.00_linux.run
$ sudo rm -rf cuda_11.1.1_455.32.00_linux.run
# Config CUDA_HOME
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
$ source ~/.bashrc
# Verify cudatoolkit installation
$ nvcc --version
# Check gcc version
$ gcc --version
# Check docker version
$ docker --version
# Check compute capacity (deviceQuery)
$ /usr/local/cuda/extras/demo_suite/deviceQuery
```

#### Build

```bash
$ cd depth_completion
$ make docker-build
```

## Training

```bash
$ cd depth_completion
$ make docker-start-interactive
$ python scripts/train.py configs/xxx.yaml
```

#### Two-stage training

```bash
# Stage 1 (w/o SAN)
$ python scripts/train.py configs/train_sparse+self_m_resnet_scania_lr.yaml
# Stage 2 (w/ SAN, update the value of checkpoint_path with a pre-trained model from stage 1)
$ python scripts/train.py configs/train_sparse+self_m_resnet_san_scania_lr.yaml
```
