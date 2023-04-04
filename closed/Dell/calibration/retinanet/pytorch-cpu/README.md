# Setup Instructions

### Install Dependencies

#### Create Anaconda environment
```
sudo apt install g++
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x Anaconda3-2020.02-Linux-x86_64.sh
bash ./Anaconda3-2020.02-Linux-x86_64.sh
```

#### Install Python and IPex
```
CUR_DIR=$(pwd)
git clone <path/to/this/repo>
cd <path/to/this/repo>/closed/Intel/calibration/retinanet/pytorch-cpu
bash prepare_calibration_env.sh
conda activate retinanet-calibration-env
```

### Download Model
```
CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
mkdir -p ${WORKLOAD_DATA}

wget --no-check-certificate 'https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth' -O 'retinanet-model.pth'
mv 'retinanet-model.pth' ${WORKLOAD_DATA}/

```

### Download (Calibration) Dataset
```
bash openimages_calibration_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages-calibration 
```


### Run Calibration

```
export CALIBRATION_DATA_DIR=${WORKLOAD_DATA}/openimages-calibration/train/data
export MODEL_CHECKPOINT=${WORKLOAD_DATA}/retinanet-model.pth
export CALIBRATION_ANNOTATIONS=${WORKLOAD_DATA}/openimages-calibration/annotations/openimages-mlperf-calibration.json
bash run_calibration.sh
```

+ Generated int8 model is saved in `${WORKLOAD_DATA}/retinanet-int8-model.pth`
