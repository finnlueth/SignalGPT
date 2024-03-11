# SignalGPT

## Abstract

SignalGPT aims to enhance the prediction of signal peptides by using large language models, specifically focusing on the Low-Rank Adaptation technique to efficiently fine-tune an array of domain expert pre-trained model adapters.
It matches the performance of previous methods while being lightweight and inexpensive to train.



## EC2 Commands
```sh
aws configure
aws configure list
# set up new key and add to aws configure

sudo yum erase nvidia cuda
sudo yum install gcc make
sudo yum install -y gcc kernel-devel-$(uname -r)
aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
chmod +x NVIDIA-Linux-x86_64*.run
sudo dnf install kernel-modules-extra
sudo /bin/sh ./NVIDIA-Linux-x86_64*.run
nvidia-smi
# nvidia-smi -q | head

sudo yum install git -y
sudo yum install python311
sudo yum update -y
nano ~/.bashrc
sudo yum -y install python-pip
pip install poetry

git config --global user.name "name"
git config --global user.email "email"

sudo dnf install kernel-modules-extra


```