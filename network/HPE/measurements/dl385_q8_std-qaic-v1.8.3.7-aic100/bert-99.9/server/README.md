# Setup
    Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.aedk/README.md).

# Benchmarking
```
SDK_VER=v1.8.3.7 POWER=no SUT=q8_std_network DOCKER=no SERVER_ONLY=yes WORKLOADS="bert" $(ck find ck-qaic:script:run)/run_edge.sh
```
