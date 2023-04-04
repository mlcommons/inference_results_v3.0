# Setup
    Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.docker/README.md).

# Benchmarking
```
SDK_VER=v1.8.3.7 POWER=no SUT=q8_std_dc DOCKER=yes SERVER_ONLY=yes WORKLOADS="retinanet" $(ck find ck-qaic:script:run)/run_datacenter.sh
```
