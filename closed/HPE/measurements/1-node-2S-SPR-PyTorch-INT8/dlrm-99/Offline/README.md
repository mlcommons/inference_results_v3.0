# C++ SUT for DLRM inference

## How to compile
### 1. Install Intel oneAPI compiler
```bash
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18679/l_HPCKit_p_2022.2.0.191.sh
sudo bash l_HPCKit_p_2022.2.0.191.sh

source /opt/intel/oneapi/compiler/2022.1.0/env/vars.sh
```

### 2. create conda environment
```bash
export WORKDIR=$PWD

source ~/anaconda3/bin/activate
conda create -n dlrm python=3.9
conda update -n base -c defaults conda --yes
conda activate dlrm
```

### 3. setup conda environment
```bash
cd ${WORKDIR}
git clone <path/to/this/repo>
ln -s <path/to/this/repo>/closed/Intel/code/dlrm-99.9/pytorch-cpu pytorch-cpu

bash ${WORKDIR}/pytorch-cpu/src/prepare_conda_env.sh
```

### 4. loadgen
```bash
cd ${WORKDIR}
git clone https://github.com/mlcommons/inference.git
cd inference/loadgen
mkdir build
cd build
CC=icx CXX=icpx cmake .. -DCMAKE_INSTALL_PREFIX=${WORKDIR}/loadgen
make -j && make install
cd ..
cp ../mlperf.conf $WORKDIR/pytorch-cpu/.
```

### 5. oneDNN
```bash
cd ${WORKDIR}
git clone -b v2.7 https://github.com/oneapi-src/oneDNN.git
cd oneDNN
mkdir build
cd build
CC=icx CXX=icpx cmake .. -DCMAKE_INSTALL_PREFIX=${WORKDIR}/spronednn
make -j && make install
```

### 6. cnpy
```bash
cd ${WORKDIR}/pytorch-cpu/src/
git clone https://github.com/rogersce/cnpy.git
cd cnpy
git checkout 4e8810b1a8637695171ed346ce68f6984e585ef4
```

### 7. C++ SUT
```bash
cd ${WORKDIR}/pytorch-cpu/src
mkdir build
cd build
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:${WORKDIR}/spronednn
CC=icx CXX=icpx cmake .. -DLOADGEN_DIR=${WORKDIR}/loadgen -DONEDNN_DIR=${WORKDIR}/spronednn
make -j
```

### 8. prepare dataset and model
```bash
# download dataset
Create a directory (such as ${WORKDIR}/dataset/terabyte_input) which contain:
	day_fea_count.npz
	terabyte_processed_test.bin

About how to get the dataset, please refer to
	https://github.com/facebookresearch/dlrm
# Note: Please generate binary dataset

# download model
# Create a directory (such as ${WORKDIR}/dataset/model):
cd ${WORKDIR}/dataset/model
wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```

### 9. preprocess dataset and model
```bash
cd ${WORKDIR}/pytorch-cpu

export MODEL=<model_dir>	# such as ${WORKDIR}/dataset/model
export DATASET=<dataset_dir>	# such as ${WORKDIR}/dataset/terabyte_input
export DUMP_PATH=<dump_out_dir>

bash dump_model_dataset.sh
```

### 10. run
```bash
cd ${WORKDIR}/pytorch-cpu

export MODEL_DIR=<dump_out_dir>
export DATA_DIR=<dump_out_dir>

# Performance mode
bash runcppsut					# offline mode
bash runcppsut performance server		# server mode
# Accuracy mode
bash runcppsut accuracy	# offline mode
bash runcppsut accuracy server

```

## Docker
### build docker and inference in docker
```bash
# Follow dataset and model preparation step above

# build docker image
git clone <path/to/this/repo>
ln -s <path/to/this/repo>/closed/Intel/code/dlrm-99.9/pytorch-cpu pytorch-cpu
# download oneAPI compiler
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18679/l_HPCKit_p_2022.2.0.191.sh
mv l_HPCKit_p_2022.2.0.191.sh <path/to/this/repo>/closed/Intel/code/dlrm-99.9
# run build docker script
cd pytorch-cpu/docker
bash build_dlrm-99.9_container.sh

# activate container
docker run --privileged --name intel_inference_dlrm -itd --net=host --ipc=host -v /data/dlrm:/data/mlperf_data/raw_dlrm  mlperf_inference_dlrm:3.0
docker exec -it intel_inference_dlrm bash

# preprocess model and dataset
cd /opt/workdir/code/dlrm/pytorch-cpu

export MODEL=<model_dir>	# such as /data/mlperf_data/raw_dlrm/model
export DATASET=<dataset_dir>	# such as /data/mlperf_data/raw_dlrm/terabyte_input
export DUMP_PATH=/data/mlperf_data/dlrm

bash dump_model_dataset.sh

# inference in docker
export MODEL_DIR=/data/mlperf_data/dlrm
export DATA_DIR=/data/mlperf_data/dlrm
# Performance mode
bash runcppsut					# offline mode
bash runcppsut performance server		# server mode
# Accuracy mode
bash runcppsut accuracy	# offline mode
bash runcppsut accuracy server
```

