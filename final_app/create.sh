conda create -y --name infosearch python==3.6.8
conda activate infosearch
conda install -f -y -q --name infosearch -c conda-forge --file conda_requirements.txt
pip install --ignore-installed --upgrade tensorflow==1.13.1
pip install -r requirements.txt

conda install -n infosearch nb_conda_kernels
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.0.56/prod/10.0_20190219/cudnn-10.0-linux-ppc64le-v7.5.0.56.tgz
sudo cp -P cuda/targets/ppc64le-linux/include/cudnn.h /usr/local/cuda-10.0/include/
sudo cp -P cuda/targets/ppc64le-linux/lib/libcudnn* /usr/local/cuda-10.0/lib64/
sudo chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*