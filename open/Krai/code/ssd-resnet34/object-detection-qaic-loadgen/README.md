# Qualcomm Cloud AI - MLPerf Inference - Object Detection (SSD Large)

Please refer to [Docker README](https://github.com/krai/ck-qaic/blob/main/docker/ssd-resnet34/README.md) for a faster way to benchmark Object Detection using SSD ResNet34 model on Qualcomm Cloud AI. The below instructions are meant only when docker environment is not used. 

## Initial System Setup

Complete the common benchmarking setup as detailed [here](https://github.com/krai/ck-qaic/blob/main/program/README.md)


<a name="prepare_coco"></a>
## Prepare the COCO 2017 validation dataset (5,000 images)

### Hint
Once you have downloaded the COCO 2017 validation dataset using CK, you can register it with CK again if needed (e.g. if you reset your CK environment) as follows:
```
ck detect soft:dataset.coco.2017.val --extra_tags=detected,full \
--full_path=/datasets/dataset-coco-2017-val/val2017/000000000139.jpg
```

<a name="prepare_coco_download"></a>
###  Download

```
ck install package --ask --tags=dataset,coco,val,2017
```


<a name="prepare_coco_preprocess"></a>
### Preprocess


**NB:** Since the preprocessed `1200x1200` COCO dataset takes up 21G, you may wish to change its destination directory by appending `--ask` to the below command.

```
ck install package --ask \
--dep_add_tags.lib-python-cv2=opencv-python-headless \
--tags=dataset,object-detection,for.ssd_resnet34.onnx.preprocessed.quantized,using-opencv,full \
--extra_tags=using-opencv
```


<a name="prepare_workload_calibrate"></a>
## Calibrate the model

### Use precalibrated profiles


```
echo "vdetected" | ck detect soft:compiler.glow.profile \
--full_path=$(ck find repo:ck-qaic)/profile/ssd-resnet34/bs.1/profile.yaml \
--extra_tags=detected,ssd_resnet34
```


### Calibrate on your own (can take hours and is optional)

**NB:** For more information, please see the [calibration README](https://github.com/krai/ck-qaic/blob/main/package/model-qaic-calibrate/README.md#ssd_resnet34).

The COCO 2017 training dataset takes `20G`. Use `--ask` to confirm the destination directory.

```
ck install package --ask --tags=dataset,coco,train,2017
```
```
ck install package --tags=dataset,coco,calibration,mlperf
```

```
ck install package --ask --tags=dataset,coco.2017,calibration,for.ssd_resnet34.onnx.preprocessed
```
```
ck install package --tags=profile,ssd_resnet34
```


<a name="prepare_workload_compile"></a>
## Compile the workload
### Compilation for 20w AEDKs (edge category)
```
ck install package --tags=model,qaic,ssd_resnet34,ssd_resnet34.aedk_20w.offline
```
```
ck install package --tags=model,qaic,ssd_resnet34,ssd_resnet34.aedk_20w.singlestream
```
```
ck install package --tags=model,qaic,ssd_resnet34,ssd_resnet34.aedk_20w.multistream
```
### Compilation for 15w AEDKs (edge category)
```
ck install package --tags=model,qaic,ssd_resnet34,ssd_resnet34.aedk_15w.offline
```
```
ck install package --tags=model,qaic,ssd_resnet34,ssd_resnet34.aedk_15w.singlestream
```
```
ck install package --tags=model,qaic,ssd_resnet34,ssd_resnet34.aedk_15w.multistream
```
Once models are compiled for AEDKs they can be installed on to the device(s) using [this](https://github.com/krai/ck-qaic/tree/main/script/setup.aedk#hr-compile-the-models-and-copy-to-the-device) script.

### Compilation for edge category 16 NSP PCIe
```
ck install package --tags=model,qaic,ssd_resnet34,ssd_resnet34.pcie.16nsp.offline
```
```
ck install package --tags=model,qaic,ssd_resnet34,ssd_resnet34.pcie.16nsp.singlestream
```
```
ck install package --tags=model,qaic,ssd_resnet34,ssd_resnet34.pcie.16nsp.multistream
```

### Compilation for datacenter category 16 NSP PCIe

```
ck install package --tags=model,qaic,ssd_resnet34,ssd_resnet34.pcie.16nsp.offline
```


## Benchmarking
For benchmarking for different System Under Tests, please see [here](https://github.com/krai/ck-qaic/blob/main/program/object-detection-qaic-loadgen/README.SSD_Large.benchmarking.md)

<a name="info"></a>
# Further info

Please contact anton@krai.ai if you have any problems or questions.
