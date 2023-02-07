@ Windos 介面
# WSL Ubuntu 介面
□ WSL Docker 介面
% Vi 介面

#####  設定 WSL #####  

@ wsl --import Ubuntu D:\WSL\Ubuntu D:\WSL\ubuntu.tar --version 2

#####  設定 systemd #####  

@ wsl -d Ubuntu
# git clone https://github.com/DamionGans/ubuntu-wsl2-systemd-script.git
# cd ubuntu-wsl2-systemd-script/
# bash ubuntu-wsl2-systemd-script.sh --force
# vi /etc/wsl.conf
% [boot]
% systemd=true
# exit
@ wsl --shutdown
@ wsl -d Ubuntu

#####  設定 openssh #####  

# apt install -y openssh-server
# vi /etc/ssh/sshd_config
# systemctl restart sshd.service
- systemctl status ssh.service

#####  設定 nvidia-smi + nvcc + nvidia-container-runtime #####  

# wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
# sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
# sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
# sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# sudo apt-get update
# sudo apt-get -y install cuda
# ls /usr/local/cuda/bin | grep nvcc
# sudo vim ~/.bashrc
% export PATH=/usr/local/cuda/bin:$PATH
% export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# source ~/.bashrc
# nvidia-smi
# nvcc --version
# curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
# sudo apt-get update
# apt-get install -y nvidia-container-runtime

#####  安裝 docker #####  

# apt install -y docker.io
# systemctl restart docker

#####  安裝 docker #####  

# docker run -itd \
	--name python39-gpu \
	--gpus all \
	--restart always \
	-e ACCEPT_EULA=Y \
	-v /Docker/Python39/Volumes/Library:/Library \
	-v /Docker/Python39/Volumes/Data:/Data \
	nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04 

# docker exec -it python39-gpu bash
□ apt update
□ apt install -y build-essential zlib1g-dev wget
□ apt install -y libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev  
□ wget -O ~/Python-3.9.13.tgz https://www.python.org/ftp/python/3.9.13/Python-3.9.13.tgz 
□ tar -xvf ~/Python-3.9.13.tgz 
□ cd /Python-3.9.13
□ ./configure --enable-optimizations 
□ make 
□ make altinstall
□ cp /usr/local/bin/python3.9 /usr/local/bin/python3
□ cp /usr/local/bin/pip3.9 /usr/local/bin/pip

- docker stop python39-gpu 
- docker rm python39-gpu 
- docker exec -it python39 python3 /Data/ScientificAnalysis/Example_P34PyTorch_02_MakeRunModel_V0_0_1.py
- docker exec -it python39-gpu python3 /Data/ScientificAnalysis/Example_P34PyTorch_02_MakeRunModel_V0_0_1.py

# docker run -itd \
	--name python39 \
	--restart always \
	-e ACCEPT_EULA=Y \
	-v /Docker/Python39/Volumes/Library:/Library \
	-v /Docker/Python39/Volumes/Data:/Data \
	vicying/python:3.9.13-cpu-0.1.0

# docker run -itd \
	--name python39-gpu \
	--gpus all \
	--restart always \
	-e ACCEPT_EULA=Y \
	-v /Docker/Python39/Volumes/Library:/Library \
	-v /Docker/Python39/Volumes/Data:/Data \
	vicying/python:3.9.13-cpu-0.1.0