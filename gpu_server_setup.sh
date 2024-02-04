for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# sudo docker pull maltel/session-rec-gpu:v1



# For docker nvidia runtime
# sudo apt install nvidia-cuda-toolkit

# sudo tee /etc/docker/daemon.json <<EOF
# {
#     "runtimes": {
#         "nvidia": {
#             "path": "/usr/bin/nvidia-container-runtime",
#             "runtimeArgs": []
#         }
#     }
# }
# EOF
# sudo pkill -SIGHUP dockerd

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
source ~/.bashrc

sudo apt install curl

$ curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install


export "$(grep -vE "^(#.*|\s*)$" .env)"

aws configure set aws_access_key_id S3_ACCESS_KEY
aws configure set aws_secret_access_key S3_SECRET_KEY
aws configure set default.region eu-north-1
aws configure set default.output json

conda activate srec37


sudo apt-get install libopenblas-dev
sudo apt install build-essential
sudo apt -y install  g++-9 
sudo apt-get install python3-pygpu 
# ~/miniconda3/bin/conda init bash

# This should be changed to the correct path
# git clone https://github.com/snorrealv/session-rec.git
# sudo chown -R ubuntu:ubuntu session-rec/
# cd session-rec
# conda env create --file environment_gpu.yml 
