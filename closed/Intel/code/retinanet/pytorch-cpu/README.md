## Setup Instructions
+ To run with docker, first setup conda enviroment, download the dataset and generate int8 model, then navigate to **Setup with docker image** 

### Setup Conda Environment and Build Dependencies
+ Download and install Anaconda3
  ```
  wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
  bash Anaconda3-2022.05-Linux-x86_64.sh
  ```
+ Setup conda environment for to install requirements, and build the src code
```
CUR_DIR=$(pwd)
git clone <path/to/this/repo>
cd <path/to/this/repo>/closed/Intel/code/retinanet/pytorch-cpu
bash prepare_env.sh
conda activate retinanet-env
```

+ Build Benchmark
```
bash build_binaries.sh
```


### Download the dataset

+ Setup env vars
```
CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
mkdir -p ${WORKLOAD_DATA}

export ENV_DEPS_DIR=${CUR_DIR}/retinanet-env
```

+ Download OpenImages (264) dataset
```
bash openimages_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages
```
Images are downloaded to `${WORKLOAD_DATA}/openimages`

+ Download Calibration images
```
bash openimages_calibration_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages-calibration
```
Calibration dataset downloaded to `${WORKLOAD_DATA}/openimages-calibration`


### Download Model
```
wget --no-check-certificate 'https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth' -O 'retinanet-model.pth'
mv 'retinanet-model.pth' ${WORKLOAD_DATA}/
```

### Calibrate and generate torchscript model

Run Calibration
```
export CALIBRATION_DATA_DIR=${WORKLOAD_DATA}/openimages-calibration/train/data
export MODEL_CHECKPOINT=${WORKLOAD_DATA}/retinanet-model.pth
export CALIBRATION_ANNOTATIONS=${WORKLOAD_DATA}/openimages-calibration/annotations/openimages-mlperf-calibration.json
bash run_calibration.sh
```

### Run Benchmark
+ exports 
```
source setup_env.sh
```
#### Performance
+ Offline
```
run_offline.sh
```

+ Server
```
run_server.sh
```

#### Accuracy
+ Offline
```
run_offline_accuracy.sh
```

+ Server
```
run_server_accuracy.sh
```

### Setup with docker image

Please make sure to finish the data and model prepations steps above.

```
# build docker image

bash build_retinanet_container.sh

# start a docker container and login

docker run --name intel_retinanet --privileged -itd --net=host --ipc=host -v ${WORKLOAD_DATA}:/opt/workdir/code/retinanet/pytorch-cpu/data mlperf_inference_datacenter_retinanet:3.0

docker exec -it intel_retinanet bash 
cd retinanet/pytorch-cpu/
export http_proxy=<your/proxy>
export https_proxy=<your/proxy>
```

### Run Benchmark
+ exports 
```
source setup_env.sh
```
#### Performance
+ Offline
```
run_offline.sh
```

+ Server
```
run_server.sh
```

#### Accuracy
+ Offline
```
run_offline_accuracy.sh
```

+ Server
```
run_server_accuracy.sh
```
