# Intel's BootstrapNAS MLPerf Retinanet Submission

## Generate BootstrapNAS INT8 model

The BootstrapNAS models were generated and optimized following these steps:
1. Starting from the FP32 PyTorch Retinanet [model](https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth), we remove a level in Retinanet's FPN and use BootstrapNAS' capabilities to generate a weight-sharing super-network.
2. Train the super-network and apply a hardware-aware search to identify high-performing subnetworks for the target hardware. Approach is described in our [paper](https://arxiv.org/abs/2112.10878).
3. Quantize (8-bit) the subnetworks using the [Neural Network Compression Framework](https://github.com/openvinotoolkit/nncf) (NNCF).
4. Convert the quantize model to be used in an Intel Extension for PyTorch (IPEX) pipeline.

​

## Model Quantization

The models were quantized (8-bit) using the following configuration for NNCF:
```python
model = torch.load(<SUBNETWORK_MODEL_PATH>)
nncf_config = {
    "input_info": {
        "sample_size": [1, 3, 800, 800]
    },
    "compression": {
        "algorithm": "quantization",
        "overflow_fix": "disable",
        "ignored_scopes": [
            "RetinaNet/__sub___0",
            "RetinaNet/__truediv___0",
            "RetinaNet/__getitem___1",
            "RetinaNet/interpolate_0",
            "RetinaNet/__getitem___2"
        ],
        "initializer": {
            "range": {
              "num_init_samples": 3840,
            },
            "batchnorm_adaptation": {
                "num_bn_adaptation_samples": 0,
            }
        },
        "weights": {
            "mode": "symmetric",
            "signed": true,
            "per_channel": false
        },
        "activations": {"mode": "symmetric"}
     },
    "target_device": "CPU"
}

nncf_config = NNCFConfig.from_dict(nncf_config)
nncf_config = register_default_init_args(
            nncf_config, data_loader, device=nncf_config.get('device'))
nncf_network = create_nncf_network(model, nncf_config)
compression_ctrl, compressed_model = create_ctrl(nncf_network, config)
```
More information about NNCF's quantization approach can be found [here](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md).

## Model Optimization and Generation of INT8 model
​
The model is converted to reference PyTorch FX and used as input in an Intel Extension for Pytorch pipeline. (This step requires PyTorch/IPEX 1.13)

## Setup Instructions

### Setup Conda Environment and Build Dependencies
+ Download and install Anaconda3
  ```bash
  wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
  bash Anaconda3-2022.05-Linux-x86_64.sh
  export CONDA_BASE_DIR=${HOME}/anaconda3
  export PATH=${PATH}:${CONDA_BASE_DIR}/bin
  ```
  Here we assume, Anaconda is installed in `${HOME}` directory.  If this is not true, please update `${CONDA_BASE_DIR}` accordingly.

+ Setup conda environment to install requirements, and build the src code.
  ```bash
  CUR_DIR=$(pwd)
  git clone <path/to/this/repo>
  cd <path/to/this/repo>/open/Intel/code/retinanet/pytorch-cpu
  export USE_CUDA=0
  bash prepare_env.sh
  conda activate retinanet-env-open
  ```

### Download the dataset

+ Setup env vars
  ```bash
  CUR_DIR=$(pwd)
  export WORKLOAD_DATA=${CUR_DIR}/data
  mkdir -p ${WORKLOAD_DATA}

  export ENV_DEPS_DIR=${CUR_DIR}/retinanet-env-open
  ```

+ Download OpenImages (264) dataset
  ```bash
  bash openimages_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages
  ```
  Images are downloaded to `${WORKLOAD_DATA}/openimages`

+ Download Calibration images
  ```bash
  bash openimages_calibration_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages-calibration
  ```
  Calibration dataset downloaded to `${WORKLOAD_DATA}/openimages-calibration`

### Calibrate and generate torchscript model

Run Calibration
```bash
cp /path/to/fx_qat.pth ${WORKLOAD_DATA}/
export CALIBRATION_DATA_DIR=${WORKLOAD_DATA}/openimages-calibration/train/data
export MODEL_CHECKPOINT=${WORKLOAD_DATA}/fx_qat.pth
export CALIBRATION_ANNOTATIONS=${WORKLOAD_DATA}/openimages-calibration/annotations/openimages-mlperf-calibration.json
bash run_calibration.sh
```
* Generated model is located in ${WORKLOAD_DATA}.  We have also provided in [./model](./model/).  To use provided model, run:
  ```bash
  cp ./model/retinanet-BNAS-int8-model.pth ${WORKLOAD_DATA}/
  ```

