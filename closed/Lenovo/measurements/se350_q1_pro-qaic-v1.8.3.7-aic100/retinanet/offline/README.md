# Setup
    Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.docker/README.md).

# Benchmarking
```
SDK_VER=v1.8.3.7 POWER=yes SUT=q1_pro_edge DOCKER=yes OFFLINE_ONLY=yes WORKLOADS="retinanet" $(ck find ck-qaic:script:run)/run_edge.sh
```
