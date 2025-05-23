# SignalGPT

## Abstract

SignalGPT aims to enhance the prediction of signal peptides by using large language models, specifically focusing on the Low-Rank Adaptation technique to efficiently fine-tune an array of domain expert pre-trained model adapters.
It matches the performance of previous methods while being lightweight and inexpensive to train.

[Slides](https://docs.google.com/presentation/d/1y0HurXYgF_IDueYM5-PA0CpQOR23i_AK1LwMXomqPaI/edit?usp=sharing)

[Report](https://www.overleaf.com/read/kzysfkwpqxrx#6bed4chttps://www.overleaf.com/read/kzysfkwpqxrx%236bed4c)

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
sudo yum install python311 -y
sudo yum update -y
nano ~/.bashrc # alias python=python3
sudo yum -y install python-pip
pip install poetry

git config --global user.name "name"
git config --global user.email "email"

ssh-keygen -t ed25519 -C "email"

sudo yum install docker
sudo usermod -a -G docker ec2-user
newgrp docker
sudo systemctl enable docker.service
sudo systemctl start docker.service
```
