# Setup with docker

<details>

# Qualcomm Cloud AI - MLPerf Inference - Datacenter and Edge servers

We provide instructions to set up an Edge appliance similar to 
Qualcomm Datacenter and Edge servers, for MLPerf Inference
benchmarking from scratch.

## General setup
Please install Collective Knowledge (ck) and the `ck-qaic` package, if it hasn't been done.
```
pip install ck
ck pull repo --url=http://github.com/krai/ck-qaic.git
```

Go to the following directory as base.
```
cd $(ck find repo:ck-qaic)/script/setup.docker
```

## Docker setup

### Host OS dependent

#### Ubuntu host (supported: Ubuntu 20.04)
```
WORKSPACE_DIR=/local/mnt/workspace bash setup_ubuntu.sh
```

#### CentOS host (supported: CentOS 7)
```
WORKSPACE_DIR=/local/mnt/workspace bash setup_centos.sh
```

**NB:** Log out and log back in for the necessary group permissions to take effect.

### Host OS independent

#### Set up Collective Knowledge environment
```
WORKSPACE_DIR=/local/mnt/workspace bash setup_ck.sh
```

### Target OS dependent, SDK dependent

#### Create Docker images

**NB:** In principle, you can use any combination of the host OS and target OS e.g. Ubuntu host and CentOS target.  For simplicity, however, we recommend to use the same OS to satisfy MLPerf requirements.

**NB:** Make sure to have copied the required datasets (e.g. ImageNet) and SDKs
to `$WORKSPACE/datasets` and `$WORKSPACE/sdks`, respectively.

Arguments:
- Use `WORKLOADS=resnet50,retinanet,ssd-resnet34,ssd-mobilenet` to select models. Default: `WORKLOADS=resnet50,retinanet`

- Use `COMPILE_PRO=yes COMPILE_STD=no` or `COMPILE_PRO=no COMPILE_STD=yes` to compile for PCIe Pro and PCIe Standard server cards, respectively.
Default: `COMPILE_PRO=yes COMPILE_STD=no`

- Use `PRECALIBRATED_PROFILE=yes` to use a precalibrated profile and `PRECALIBRATED_PROFILE=no` to calibrate the workload from scratch. Default: `PRECALIBRATED_PROFILE=yes`
```
WORKLOADS=ssd-resnet34,ssd-mobilenet COMPILE_PRO=no COMPILE_STD=yes PRECALIBRATED_PROFILE=no DOCKER_OS=ubuntu SDK_DIR=/data/qaic/1.8.0.137 SDK_VER=1.8.0.137 TIMEZONE=Europe/London bash setup_images.sh
```

#### Test Docker images

##### Edge - Q1 Pro

```
cd $(ck find ck-qaic:script:run)
QUICK_RUN=yes SDK_VER=1.7.1.12 DOCKER=yes SUT=r282_z93_q1_prev ./run_edge.sh
```

<details><pre>
$ ck list $CK_EXPERIMENT_REPO:experiment:*r282_z93_q1_prev*resnet50* | wc -l
6
$ ck list $CK_EXPERIMENT_REPO:experiment:*r282_z93_q1_prev*bert* | wc -l
4
$ grep "accuracy\":\ 7" $CK_EXPERIMENT_DIR/*r282_z93_q1_prev*/*.0001.json -Rh
        "accuracy": 75.956,
        "accuracy": 75.956,
        "accuracy": 75.956,
$ grep \"f1\" $CK_EXPERIMENT_DIR/*r282_z93_q1_prev*/*.0001.json -Rh
        "f1": 90.22951222279839,
        "f1": 90.08969847302875,
$ grep "Samples per second:" $CK_EXPERIMENT_DIR/*r282_z93_q1_prev*target_qps.1*/*.0001.json -Rh
            "Samples per second: 658.248\n",
            "Samples per second: 21903.1\n",
$ grep "Early stopping 90th percentile estimate:" $CK_EXPERIMENT_DIR/*r282_z93_q1_prev*target_latency.1000*/*.0001.json -Rh | grep -v MLLOG
            " * Early stopping 90th percentile estimate: 13456661\n",
            " * Early stopping 90th percentile estimate: 611977\n",
$ grep "99th percentile latency (ns) :" $CK_EXPERIMENT_DIR/*r282_z93_q1_prev*target_latency.1000*/*.0001.json -Rh
            "99th percentile latency (ns) : 1842326\n",
</pre></details>

### Further info

#### Current workloads

1. [Image Classification](https://github.com/krai/ck-qaic/tree/main/docker/resnet50)
1. [Natural Language Processing](https://github.com/krai/ck-qaic/blob/main/docker/bert/README.md)
2. [Object Detection RetinaNet](https://github.com/krai/ck-qaic/blob/main/docker/retinanet/README.md)

#### Deprecated workloads

1. [Object Detection Small](https://github.com/krai/ck-qaic/tree/main/docker/ssd-mobilenet)
1. [Object Detection Large](https://github.com/krai/ck-qaic/tree/main/docker/ssd-resnet34)

# Info

Please contact anton@krai.ai if you have any problems or questions.

</details>

<br>
<br>
<br>

# Setup directly on AEDK

<details>

# Qualcomm Cloud AI - MLPerf Inference - Edge Devices (AEDK)

We provide instructions to set up an Edge appliance similar to  Qualcomm Edge
AI Development Kit (AEDK), which we call "the device", for MLPerf Inference
benchmarking from scratch.

We assume that the user operates a Linux workstation (or a Windows laptop
under WSL), which we call "the host". We further assume that the host has
installed the Collective Knowledge framework (CK) and the QAIC Apps SDK
matching the QAIC Platform SDK to be installed on the device.

Instructions below alternate between running on the host (marked with `H`)
and on the device (marked with `D`). Instructions to be run as superuser are
additionally marked with `S`.

Some instructions are to be run only once (marked with `1`). Some instructions
are to be repeated as needed e.g. for new SDK versions (marked with `R`).

# A. Initial host setup

## `[H1]` General setup
Please install Collective Knowledge (ck) and the `ck-qaic` package, if it hasn't been done.
```
pip install ck
ck pull repo --url=http://github.com/krai/ck-qaic.git
```

Go to the following directory as base.
```
cd $(ck find repo:ck-qaic)/script/setup.aedk
```

## `[H1]` Set variables and paths
Update device name, paths and variables in `config.sh`, then `source` it
```
source $(ck find repo:ck-qaic)/script/setup.aedk/config.sh
```

**NB:** The full installation can take more than 50G. If the space on the root
partition of the device is limited and you wish to use a different partition,
change the `DEVICE_BASE_DIR` in `config.sh`.

# B. Initial device setup under the `root` user

## `[H1]` Connect to the device as `root`
Connect to the device as `root` e.g.:
```
ssh -p ${DEVICE_PORT} root@${DEVICE_IP}
```

## `[D1]` Clone the repository with setup scripts

```
git clone https://github.com/krai/ck-qaic /tmp/ck-qaic
```

## `[D1S]` Run
Go to the temporary directory:
```
cd /tmp/ck-qaic/script/setup.aedk
```

Check the config file:
```
cat ./config.sh
```

<details><pre>
#!/bin/bash

export DEVICE_IP=aedk3
export DEVICE_PORT=3233
export DEVICE_BASE_DIR="/home"
export DEVICE_GROUP=krai
export DEVICE_USER=krai
export DEVICE_OS=ubuntu
export DEVICE_OS_OVERRIDE=no
export DEVICE_DATASETS_DIR="${DEVICE_BASE_DIR}/${DEVICE_USER}"
export HOST_DATASETS_DIR="/datasets"
export PYTHON_VERSION=3.9.14
export TIMEZONE="US/Central"
export INSTALL_BENCHMARK_RESNET50=yes
export INSTALL_BENCHMARK_BERT=yes
</pre></details>

Source it if you are happy with the settings and run:
```
source ./config.sh && ./1.run_as_root.sh
```

Alternatively, you can override variables from the command line e.g.:
```
time DEVICE_BASE_DIR=/data TIMEZONE=London/Europe ./1.run_as_root.sh
```

<details><pre>
root@aus655-gloria-1:~# df -h /home
Filesystem      Size  Used Avail Use% Mounted on
/dev/root        99G   11G   89G  11% /
root@aus655-gloria-1:~# df -h /datasets
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  880G   77M  835G   1% /datasets
root@aus655-gloria-1:/tmp/ck-qaic/script/setup.aedk# time DEVICE_BASE_DIR=/datasets TIMEZONE=US/Central ./1.run_as_root.sh
...
Sat Jul 23 09:05:56 CDT 2022
real    3m42.599s
user    6m4.276s
sys     1m5.008s
</pre></details>

## `[D1S]` Set user password
```
passwd ${DEVICE_USER}
```

# C. Initial device setup under the `krai` user

## `[H1]` Connect to the device as `krai`
Connect to the device as `krai` e.g.:
```
ssh -p ${DEVICE_PORT} krai@${DEVICE_IP}
```

## `[D1]` Update scripts permissions
```
sudo chown -R krai:krai /tmp/ck-qaic
sudo chmod u+x /tmp/ck-qaic/script/setup.aedk/*.sh
```

## `[D1]` Run
```
cd /tmp/ck-qaic/script/setup.aedk
source ./config.sh && time ./2.run_as_krai.sh
source ~/.bashrc
source ./config.sh && time ./3.run_as_krai.sh
```

# D. Set up ImageNet and other datasets

Suppose the ImageNet validation dataset (50,000 images) is in an archive (6.4G) called
`dataset-imagenet-ilsvrc2012-val.tar` in the `${HOST_DATASETS_DIR}` on the host machine.
Validate the `md5sum` checksum.

<details><pre>
&dollar; md5sum ${HOST_DATASETS_DIR}/dataset-imagenet-ilsvrc2012-val.tar
3f31a40f2bb902e28aa23aad0fc8e383  dataset-imagenet-ilsvrc2012-val.tar
</pre></details>

<details><pre>
krai@aus655-gloria-1:/datasets&dollar; md5sum imagenet.tar
2398abe8c17b3bf5df61946fff0b8494  imagenet.tar
</pre></details>


## `[H1]` Copy the ImageNet dataset from the host to the device
```
scp -P ${DEVICE_PORT} ${HOST_DATASETS_DIR}/dataset-imagenet-ilsvrc2012-val.tar root@${DEVICE_IP}:${DEVICE_DATASETS_DIR}
```

## `[D1]` Extract and preprocess ImageNet on the device
```
cd /tmp/ck-qaic/script/setup.aedk
source ./config.sh && time ./4.install_workloads.sh
```

<details>

ResNet50 example:
<pre>
krai@aus655-gloria-1:/tmp/ck-qaic/script/setup.aedk&dollar; time INSTALL_WORKLOAD_RESNET50=yes INSTALL_WORKLOAD_BERT=no DEVICE_DATASETS_DIR=/datasets DEVICE_IMAGENET_DIR=imagenet ./4.install_workloads.sh
...
real    10m3.297s
user    8m24.348s
sys     12m31.936s
</pre>

BERT example:
<pre>
krai@aus655-gloria-1:/tmp/ck-qaic/script/setup.aedk&dollar; time INSTALL_WORKLOAD_RESNET50=no INSTALL_WORKLOAD_BERT=yes ./4.install_workloads.sh
...
real    15m10.001s
user    27m41.982s
sys     2m2.424s
</pre>

</details>

Detect Open Images
<details><pre>
cp -r $(ck find script:setup.aedk)/593d9d6728534c67 ~/CK/local/env/ \
&& ck show env --tags=dataset,openimages,preprocessed
</pre></details>

# E. Set up QAIC SDKs
Obtain a pair of QAIC SDKs:
- Apps SDK to be used on the host for compilation (e.g. `qaic-apps-1.7.1.12.zip`).
- Platform SDK to be used on the device for execution (e.g. `qaic-platform-sdk-1.7.1.12.zip`).

These steps are to be repeated for each new SDK version (`SDK_VER` below).

## `[HSR]` Uninstall/Install the Apps SDK

Specify `SDK_DIR`, the path to a directory with one or more Apps SDK archives, and `SDK_VER`, the Apps SDK version.
The full path to the Apps SDK archive is formed as follows: `APPS_SDK=$SDK_DIR/qaic-apps-$SDK_VER.zip`.

```
SDK_DIR=/local/mnt/workspace/sdks SDK_VER=1.7.1.12 $(ck find ck-qaic:script:setup.aedk)/install_apps_sdk.sh
```

Alternatively, specify `APPS_SDK`, the full path to the Apps SDK archive.

<details><pre>
&dollar; grep build_id /opt/qti-aic/versions/apps.xml -B1
                &lsaquo;base_version&rsaquo;1.6&lsaquo;&sol;base_version&rsaquo;
                &lsaquo;build_id&rsaquo;80&lsaquo;&sol;build_id&rsaquo;
</pre></details>

## `[HR]` Copy the Platform SDK to the device

Go to the directory containing your Platform SDK archive e.g. `/local/mnt/workspace/sdks`
and copy it to the device e.g. with the `${DEVICE_IP}` address and `${DEVICE_PORT}` port:

```
export SDK_VER=1.7.1.12
scp -P ${DEVICE_PORT} qaic-platform-sdk-${SDK_VER}.zip ${DEVICE_USER}@${DEVICE_IP}:${DEVICE_BASE_DIR}/${DEVICE_USER}
```

## `[DSR]` Uninstall/Install the Platform SDK

Specify `SDK_DIR`, the path to a directory with one or more Platform SDK archives, and `SDK_VER`, the Platform SDK version.
The full path to the Platform SDK archive is formed as follows: `PLATFORM_SDK=$SDK_DIR/qaic-platform-sdk-$SDK_VER.zip`.

```
SDK_DIR=${DEVICE_BASE_DIR}/${DEVICE_USER} SDK_VER=1.7.1.12 bash $(ck find ck-qaic:script:setup.aedk)/install_platform_sdk.sh
```

Alternatively, specify `PLATFORM_SDK`, the full path to the Platform SDK archive.

<details><pre>
SDK_DIR=~ SDK_VER=1.7.1.12 $(ck find ck-qaic:script:setup.aedk)/install_platform_sdk.sh
</pre></details>

## `[HR]` Compile the workloads on the host and copy to the device

The easiest way to install the workloads to the device is to use Docker images [prebuilt](https://github.com/krai/ck-qaic/blob/main/script/setup.docker/README.md) on the host.

Build the docker image on the host. E.g.:
```
REPOSITORY                          TAG               IMAGE ID       CREATED         SIZE
krai/mlperf.bert                    ubuntu_1.7.1.12   e4f31177975f   3 weeks ago     15.2GB
```

Run the installation script.
```
cd $(ck find ck-qaic:script:setup.aedk)
SDK_VER=1.7.1.12 DEVICE_IP=192.168.0.12 DEVICE_PORT=1234 DEVICE_TYPE=aedk_15w DEVICE_PASSWORD=12345678 ./install_to_aedk.sh
```

If you do not wish to use Docker images for some reason, you can follow these instructions: 

<details>
Follow the common [instructions](https://github.com/krai/ck-qaic/blob/main/program/README.md), and then instructions for individual workloads:

1. [Image Classfication](https://github.com/krai/ck-qaic/blob/main/program/image-classification-qaic-loadgen/README.md) (ResNet50)
1. [Object Detection](https://github.com/krai/ck-qaic/blob/main/program/object-detection-qaic-loadgen/README.md) (RetinaNet)
1. [Language Processing](https://github.com/krai/ck-qaic/blob/main/program/packed-bert-qaic-loadgen/README.md) (BERT)
</details>

# F. Expected Results from Set up QAIC SDKs
If you followed the instructions in above [section](#e-set-up-qaic-sdks), you should expect to see something like this.

<details>
On Host:
<pre>
krai@aus655-el-01-5:/local/mnt/workspace/krai/CK-REPOS/ck-qaic/script/setup.aedk&dollar; time DEVICE_BASE_DIR=/datasets DEVICE_IP=10.222.147.222 DEVICE_PASSWORD=123 SDK_VER=1.7.1.12 ./install_to_aedk.sh
...
DONE (installing workloads).
real    6m58.061s
user    0m3.306s
sys     0m2.730s
</pre>

<pre>
krai@aus655-el-01-5:/local/mnt/workspace/krai/CK-REPOS/ck-qaic/script/setup.aedk&dollar; time DEVICE_BASE_DIR=/home DEVICE_IP=10.222.147.231 DEVICE_PASSWORD=123 SDK_VER=1.7.1.12 ./install_to_aedk.sh
...
DONE (installing workloads).
</pre>

On Device:

<pre>
krai@aus655-gloria-1:~&dollar; ck show env --tags=qaic,model
Env UID:         Target OS: Bits: Name:                   Version: Tags:
7f20244d5a912e91   linux-64    64 Qualcomm Cloud AI model 1.7.1.12 64bits,bs.1,calibrated-by-qaic,compiled,compiled-by-qaic,converted,host-os-linux-64,image-classification,model,qaic,qualcomm,qualcomm-ai,qualcomm-cloud-ai,resnet50,resnet50.aedk_15w.singlestream,target-os-linux-64,v1,v1.7,v1.7.1,v1.7.1.12
57f5445dc811641f   linux-64    64 Qualcomm Cloud AI model 1.7.1.12 64bits,bert,bert-99,bert-99.aedk_15w.singlestream,bert.mixed,calibrated-by-qaic,compiled,compiled-by-qaic,converted,host-os-linux-64,model,pcv.9980,qaic,qualcomm,qualcomm-ai,qualcomm-cloud-ai,quantization.calibration,seg.384,target-os-linux-64,v1,v1.7,v1.7.1,v1.7.1.12
46d8b9edf850be49   linux-64    64 Qualcomm Cloud AI model 1.7.1.12 64bits,bs.1,calibrated-by-qaic,compiled,compiled-by-qaic,converted,host-os-linux-64,image-classification,model,qaic,qualcomm,qualcomm-ai,qualcomm-cloud-ai,resnet50,resnet50.aedk_15w.multistream,target-os-linux-64,v1,v1.7,v1.7.1,v1.7.1.12
886152f267c43908   linux-64    64 Qualcomm Cloud AI model 1.7.1.12 64bits,bert,bert-99,bert-99.aedk_15w.offline,bert.mixed,calibrated-by-qaic,compiled,compiled-by-qaic,converted,host-os-linux-64,model,pcv.9980,qaic,qualcomm,qualcomm-ai,qualcomm-cloud-ai,quantization.calibration,seg.384,target-os-linux-64,v1,v1.7,v1.7.1,v1.7.1.12
8ebc64fba89f7665   linux-64    64 Qualcomm Cloud AI model 1.7.1.12 64bits,bs.8,calibrated-by-qaic,compiled,compiled-by-qaic,converted,host-os-linux-64,image-classification,model,qaic,qualcomm,qualcomm-ai,qualcomm-cloud-ai,resnet50,resnet50.aedk_15w.offline,target-os-linux-64,v1,v1.7,v1.7.1,v1.7.1.12
</pre>

</details>

# G. Benchmarking
## `[D]` Verify with a quick run
```
QUICK_RUN=yes UPDATE_CK_QAIC=no WORKLOADS=bert SDK_VER=1.7.1.12 SUT=eb6 ./run_edge.sh
```


---

# Appendix: Arguments

#### `SDK_VER`

The SDK version.
Must be set.

#### `CONTAINER_ID`

If set, use this container to compile and install workloads from, do not start a new one.
Only works with a single workload at a time e.g. `WORKLOADS=resnet50`.

<details><pre>
CONTAINER_ID=&dollar;(docker run -dt --rm krai/mlperf.resnet50.full:ubuntu_1.7.1.12 bash)
CONTAINER_ID=$CONTAINER_ID DEVICE_IP=192.168.0.12 DEVICE_PASSWORD=123 SDK_VER=1.7.1.12 ./install_to_aedk.sh
</pre></details>

#### `DEVICE_IP`

The IP address or hostname of the device.
Must be set.

#### `DEVICE_PORT`

The SSH port on the device.
Default: `22`.

#### `DEVICE_PASSWORD`

The password on the device.
Must be set.
Does not get cached.

#### `DEVICE_BASE_DIR`

The root of the user directories on the device.
Default: `/home`.

#### `DEVICE_USER`

The username on the device.
Default: `krai`.

#### `DEVICE_TYPE`

The type of the device.
Default: `aedk_15w` (e.g. for Foxconn Gloria and Alibaba Haishen).

#### `WORKLOADS`

A comma-separated list of workloads to compile and install.
Default: `WORKLOADS="resnet50,bert"`. 

#### `OFFLINE_ONLY`

Default: `OFFLINE_ONLY=no`. If `OFFLINE_ONLY=yes`, only compile and install workloads for the Offline scenario.

#### `SINGLESTREAM_ONLY`

Default: `SINGLESTREAM_ONLY=no`. If `SINGLESTREAM_ONLY=yes`, only compile and install workloads for the Single Stream scenario.

#### `MULTISTREAM_ONLY`

Default: `MULTISTREAM_ONLY=no`. If `MULTISTREAM_ONLY=yes`, only compile and install workloads for the Multi Stream scenario.

#### `UPDATE_CK_QAIC`

Default: `UPDATE_CK_QAIC=yes`. If `UPDATE_CK_QAIC=no`, do not update the `ck-qaic` repo.

#### `DRY_RUN`

Default: `DRY_RUN=no`. If `DRY_RUN=yes`, only print commands but do not execute them.

#### `DRY_COMPILE`

Default: `DRY_COMPILE=no`.
If `DRY_COMPILE=yes`, only print compilation commands.
This requires operating with workload binaries baked into the Docker image.
See `DOCKER_DEVICE_TYPE`.

#### `DRY_INSTALL`

Default: `DRY_INSTALL=no`.
If `DRY_INSTALL=yes`, only print installation commands.

#### `DOCKER`

Default: `yes`.
Whether to use Docker images to run compile and install workloads.

#### `DOCKER_OS`

Default: `DOCKER_OS=ubuntu`.
If `DOCKER_OS=ubuntu`, assume Ubuntu 20.04 based images have been created.
If `DOCKER_OS=centos`, assume CentOS 7 based images have been created.

#### `DOCKER_DEVICE_TYPE`

Default: `DOCKER_DEVICE_TYPE=pcie.16nsp`.
See `DRY_COMPILE`.

Use `DOCKER_DEVICE_TYPE=pcie.14nsp` if images have been compiled with the `COMPILE_PRO=no COMPILE_STD=yes` flags.

# Appendix: Info

Please contact anton@krai.ai if you have any problems or questions.


</details>