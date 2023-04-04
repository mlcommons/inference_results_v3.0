## Setup Instructions - Baremetal

### REQUIRMENTS:
+ GCC > 10.1
+ NUMACTL
    + CentOS:
    ```
    sudo yum install numactl-devel
    sudo yum install numactl
    ```
    
    + Ubuntu:
    ```
    sudo apt install numactl-dev
    ```

### Setup Conda Environment
+ Download and install Anaconda3
  ```
  wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
  bash Anaconda3-2022.05-Linux-x86_64.sh
  ```
+ Setup conda environment to install requirements, and build packages
  ```
  bash prepare_env.sh
  ```



### Download Imagenet Dataset for Calibration
Download ImageNet (50000) dataset
```
bash download_imagenet.sh
```
Prepare calibration 500 images into folders 
```
bash prepare_calibration_dataset.sh
```

### Download Model
```
bash download_model.sh
```
The downloaded model will be saved as ```resnet50-fp32-model.pth```

### Quantize Torchscript Model and Check Accuracy 
+ Set the following paths:
```
export DATA_CAL_DIR=calibration_dataset
export CHECKPOINT=resnet50-fp32-model.pth
```
+ Generate scales and models
```
bash generate_torch_model.sh
```

The *start* and *end* parts of the model are also saved (respectively named) in ```models```

### Build 
```
bash build_binaries.sh
```
Please follow the [instructions](#run-benchmark-common-for-docker--baremetal) in the end of the page to run the benchmark. 

## Setup Instructions - Docker 

The docker container can be created either by building it using the Dockerfile or pulling the image from Dockerhub (if available). Please download the Imagenet dataset on the host system before starting the container.

### (a) Build & Run Docker container from Dockerfile
```
cd docker/

bash build_resnet50_contanier.sh

docker run -v </path/to/ILSVRC2012_img_val>:/opt/workdir/code/resnet50/pytorch-cpu/ILSVRC2012_img_val -itd --privileged <docker image ID> /bin/bash

cd code/resnet50/pytorch-cpu
```

### (b) Pull Docker Image from Dockerhub & Run container
```
[TBD]
docker pull intel/intel-optimized-pytorch:mlperf-inference-3.0-resnet50

docker run --name intel_resnet50 -v </path/to/ILSVRC2012_img_val>:/opt/workdir/code/resnet50/pytorch-cpu/ILSVRC2012_img_val --privileged -itd --net=host \
  --ipc=host intel/intel-optimized-pytorch:mlperf-inference-3.0-resnet50:3.0
  
docker exec -it intel_resnet50 bash

cd code/resnet50/pytorch-cpu

```
### Prepare Calibration Dataset & Download Model ( Inside Container )

If you need a proxy to access the internet, replace your host proxy with the proxy server for your environment. If no proxy is needed, you can skip this step:
```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```
Prepare calibration 500 images into folders 
```
bash prepare_calibration_dataset.sh
```
Download the model
```
bash download_model.sh
```
The downloaded model will be saved as ```resnet50-fp32-model.pth```


### Quantize Torchscript Model and Check Accuracy 
+ Set the following paths:
```
export DATA_CAL_DIR=calibration_dataset
export CHECKPOINT=resnet50-fp32-model.pth
```
+ Generate scales and models
```
bash generate_torch_model.sh
```

The *start* and *end* parts of the model are also saved (respectively named) in ```models```


## Run Benchmark (Common for Docker & Baremetal)

```
export DATA_DIR=${PWD}/ILSVRC2012_img_val
export RN50_START=models/resnet50-start-int8-model.pth
export RN50_END=models/resnet50-end-int8-model.pth
export RN50_FULL=models/resnet50-full.pth
```

### Performance
+ Offline
```
bash run_offline.sh <batch_size>
```

+ Server
```
bash run_server.sh
```

### Accuracy
+ Offline
```
bash run_offline_accuracy.sh <batch_size>
```

+ Server
```
bash run_server_accuracy.sh
```

