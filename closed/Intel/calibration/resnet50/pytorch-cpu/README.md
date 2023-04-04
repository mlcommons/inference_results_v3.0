# ResNet50 Int8 Quantization Steps

## Setup Instructions

### Setup Conda Environment
+ Download and install Anaconda3
  ```
  wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
  bash Anaconda3-2022.05-Linux-x86_64.sh
  ```
+ Setup conda environment for to install requirements 
  ```
  bash prepare_calibration_env.sh
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

The generated full torchscript model is saved in ```models/resnet50-int8-model.pth```.

The *start* and *end* parts of the model are also saved (respectively named) in ```models```
