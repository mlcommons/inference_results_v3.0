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
# Example Command related to this Scenario and Workload
```
SDK_VER=v1.8.3.7 POWER=no SUT=g292_z43_q18 DOCKER=yes OFFLINE_ONLY=yes WORKLOADS="bert" $(ck find ck-qaic:script:run)/run_datacenter.sh
```
