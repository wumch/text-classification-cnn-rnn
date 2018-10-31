
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt-get update
apt-get install cuda

/opt/miniconda/bin/pip install --upgrade pip
/opt/miniconda/bin/conda install numpy scipy scikit-learn tensorflow-gpu
/opt/miniconda/bin/pip install jieba word2vec xgboost
