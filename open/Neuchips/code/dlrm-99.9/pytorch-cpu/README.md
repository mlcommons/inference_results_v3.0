# Setup from Source

<!-- ## System requirements -->
<!-- ### 1. HW requirements
| HW  |      Configuration      |
| --  | ----------------------- |
| CPU | SPR-6 @ 2 sockets/Node  |
| DDR | 512G/socket @ 3200 MT/s |
| SSD | 1 SSD/Node @ >= 1T      |

### 2. SW requirements
| SW       | Version |
|----------|---------|
| GCC      |  11.2   |
| Binutils | >= 2.35 | -->
## System Dependencies

The system for running the inference is based on following configurations: 

|  Item | Description  |
| ------------ | ------------ |
|  **Server machine** | NEUCHIPS-DLRM-AE01 |
|  **Host processor** | [AMD EPYC 9004 Series ](https://www.amd.com/en/processors/epyc-9004-series)  |
|  **Storage space** |  more than 1TB |
|  **Memory size** |  more than 384 GB |
|  **Operating system** |  Ubuntu 22.04.1 |
|  **Linux kernel** |  v5.18.17 |
|  **Compiler** |  [AOCC v4.0](https://www.amd.com/en/developer/aocc.html)|


## Steps to run NEUCHIPS-DLRM

### 1. Install anaconda 3.0
```
  mkdir <workfolder>
  cd <workfolder>
  wget -c https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ./anaconda3.sh -b -p ./anaconda3
  export WORKDIR=$PWD
  export PATH=$WORKDIR/anaconda3/bin:$PATH
  conda create -n dlrm python=3.9
  conda update -n base -c defaults conda --yes
  conda activate dlrm
```
### 2. Download Repo for DLRM MLPerf inference
```
  git clone <path/to/this/repo> 
```
### 3. Install conda dependency packages
```
  cp dlrm_pytorch/prepare_conda_env.sh .
  bash ./prepare_conda_env.sh
```
### 4. Prepare AOCC v4.0

Please refer to: [https://www.amd.com/en/developer/aocc.html](https://www.amd.com/en/developer/aocc.html)

### 5. Install mlperf loadgen 
Please refer to:  [https://github.com/mlcommons/inference/tree/master/loadgen](https://github.com/mlcommons/inference/tree/master/loadgen)
### 6. Prepare DLRM dataset and code
(1) Prepare DLRM dataset
```
   Create a directory (such as /data/mlperf_data/dlrm/) which contain:
     day_fea_count.npz
     terabyte_processed_test.bin

   About how to get the dataset, please refer to
      https://github.com/facebookresearch/dlrm
```
(2) Prepare pre-trained DLRM model
```
   cd /data/mlperf_data/dlrm/
   wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```
Please note that due to some constraints, AUC number might not be exactly the same as NEUCHIPS submit, please contact NEUCHIPS for more if necessary.
### 7. Run command for server and offline mode

(1) cd dlrm_pytorch

(2) configure DATA_DIR and MODEL_DIR #you can modify the setup_dataset.sh, then 'source ./setup_dataset.sh'
```
   export DATA_DIR=           # the path of dataset, for example as /data/mlperf_data/dlrm/
   export MODEL_DIR=          # the path of pre-trained model, for example as /data/mlperf_data/dlrm/
```
(3) configure offline mode option # currenlty used option is in setup_env_offline.sh, You can modify it, then 'source ./setup_env_offline.sh'
```
   export NUM_SOCKETS=        # i.e. 4
   export CPUS_PER_SOCKET=    # i.e. 48
   export CPUS_PER_PROCESS=   # i.e. 12. which determine how many cores for one processe running on one socket
                              #   process_number = $CPUS_PER_SOCKET / $CPUS_PER_PROCESS
   export CPUS_PER_INSTANCE=  # i.e. 12. which determine how many cores used for one instance inside one process
                              #   instance_number_per_process = $CPUS_PER_PROCESS / CPUS_PER_INSTANCE
                              #   total_instance_number_in_system = instance_number_per_process * process_number
```
(5) Disable SMT and set system to performance mode
   echo off  > /sys/devices/system/cpu/smt/control  # disable SMT
   sudo ./set_perf.sh           # set system to performance mode
```
(6) command line
   Please update setup_env_offline.sh and user.conf according to your platform resource.
   bash run_mlperf.sh --mode=offline --type=<perf/acc> --dtype=int8
```
  





