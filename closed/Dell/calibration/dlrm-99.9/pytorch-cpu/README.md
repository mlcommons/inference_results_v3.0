# Steps to run DLRM calibration

### 1. Install anaconda 3.0
```
  mkdir <workfolder>
  cd <workfolder>
  wget -c https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ./anaconda3.sh -b -p ./anaconda3
  export WORKDIR=$PWD
  export PATH=$WORKDIR/anaconda3/bin:$PATH
  conda create -n dlrm_calibration python=3.9
  conda update -n base -c defaults conda --yes
  source activate dlrm_calibration
```
### 2. Download Repo for DLRM MLPerf inference
```
  git clone <path/to/this/repo>
  ln -s <path/to/this/repo>/closed/Intel/code/dlrm-99.9/pytorch-cpu dlrm_pytorch
```
### 3. Install conda dependency packages
```
  cp dlrm_pytorch/prepare_conda_env.sh .
  bash ./prepare_conda_env.sh
```
### 4. Prepare GCC11
```
  If system has GCC11 installed, you can try to update GCC version by using:
    source /opt/rh/gcc-toolset-11/enable
  Or you can install GCC11.2 through conda by using following scripts:
    source ./setup_gcc_env.sh #install and setup GCC11 env
```
### 5. Install mlperf loadgen and intel extension for pytorch
```
  cp <path/to/this/repo>/closed/Intel/calibration/dlrm-99.9/pytorch-cpu/prepare_env.sh .
  bash prepare_env.sh
```
### 6. Prepare DLRM dataset and model
(1) Prepare Calibration dataset
```
   For calibration, need the following two files:
     day_fea_count.npz
     terabyte_processed_val.bin

   For DLRM MLPerf Int8 Inference, we use the first 128000 rows (user-item pairs) of the second half of day_23 as the calibration set.
   terabyte_processed_val.bin is the second part of day_23 which bin rows is start from the 89137319-th row of day_23.

   About how to get the day_fea_cout.npz and terabyte_processed_val.bin, please refer to
      https://github.com/facebookresearch/dlrm
```
(2) Prepare pre-trained DLRM model
```
   cd dataset
   wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```
### 7. Run command to do calibration
(1) cd <path/to/this/repo>/closed/Intel/calibration/dlrm-99.9/pytorch-cpu
(2) configure DATA_DIR and MODEL_DIR #you can modify the setup_dataset.sh, then 'source ./setup_dataset.sh'
(3) command line
```
   # do calibration
   bash ./run_calibrate.sh # run for int8 calibration
   #calibration output is int8_configure.json under output/
```
(4) based on ./output/int8_configure.json modify the scales on <path/to/this/repo>/closed/Intel/code/dlrm-99.9/pytorch-cpu/src/harness/dlrm_main.cpp:75