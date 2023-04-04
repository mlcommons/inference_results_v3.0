# Setup
    Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.docker/README.md).

# Benchmarking
```
SDK_VER=v1.8.3.7 POWER=no SUT=q3_std_edge DOCKER=yes MULTISTREAM_ONLY=yes WORKLOADS="resnet50" $(ck find ck-qaic:script:run)/run_edge.sh
```
