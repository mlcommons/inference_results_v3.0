# RNN-T MLPerf Inference BKC

## HW & SW requirements
###
```
  SPR 2 sockets
  GCC >= 11
```

## Steps to run RNN-T with three options

### Option 1: Run on bare metal
#### 1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ~/anaconda3.sh -b -p ~/anaconda3
  export PATH=~/anaconda3/bin:$PATH
```

#### 2. End-to-end run inference
Execute `run.sh`. The end-to-end process including:
| STAGE(default -2) | STEP |
|  -  | -  |
| -2 | Prepare conda environment |
| -1 | Prepare environment |
| 0 | Download model |
| 1 | Download dataset |
| 2 | Pre-process dataset |
| 3 | Calibration |
| 4 | Build model |
| 5 | Run Offline/Server accuracy & benchmark |

You can also use the following command to start with your custom conda-env/work-dir/step.
```
  [CONDA_ENV] [WORK_DIR] [STAGE] bash run.sh
```

### Option 2: Build docker container
```
  cd docker
  bash build_rnnt-99_container.sh
  docker run --name intel_rnnt --privileged -itd -v /data/mlperf_data:/data/mlperf_data --net=host --ipc=host mlperf_inference_rnnt:3.0
  docker ps -a #get container "id"
  docker exec -it <id> bash
  cd /opt/workdir/code/rnnt/pytorch-cpu
  SKIP_BUILD=1 STAGE=0 bash run.sh
```
