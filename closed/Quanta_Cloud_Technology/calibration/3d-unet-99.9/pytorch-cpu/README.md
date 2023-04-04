# MLPerf Inference Benchmarks for Medical Image 3D Segmentation

## Steps to run 3D-UNet calibration


### 0. HW and SW requirements
```
  SPR 2 sockets
  GCC >= 11.2
```

### 1. Install Anaconda 3.0
```
  mkdir <workfolder>
  cd <workfolder>
  wget -c https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ./anaconda3.sh -b -p ./anaconda3
  export WORKDIR=$PWD
  export PATH=$WORKDIR/anaconda3/bin:$PATH
  conda create -n 3dunet python=3.8
  conda update -n base -c defaults conda --yes
  source activate 3dunet
```

### 2. Download Dataset
```
  cd ~
  mkdir -p mlperf_data/3dunet-kits
  cd mlperf_data/3dunet-kits
  git clone https://github.com/neheller/kits19
  cd kits19
  pip3 install -r requirements.txt
  python3 -m starter_code.get_imaging
  cd ~
```

### 3. Download Repo for 3D-UNet MLPerf inference and setup env
```
  git clone <this/repo>
  // REPO_PATH=<path/to/this/repo>
  bash REPO_NAME/closed/Intel/code/3d-unet-99.9/pytorch-cpu/prepare_env.sh --code=REPO_PATH
```

### 4. Run command for accuracy and performance
```
  cd REPO_NAME/closed/Intel/code/3d-unet-99.9/pytorch-cpu
  python calibrate.py
```
