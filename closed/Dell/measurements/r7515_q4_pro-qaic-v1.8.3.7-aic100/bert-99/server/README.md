# Setup
    Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.docker/README.md).

# Benchmarking
```
SDK_VER=v1.8.3.7 POWER=yes SUT=q4_pro_dc DOCKER=yes SERVER_ONLY=yes WORKLOADS="bert" $(ck find ck-qaic:script:run)/run_datacenter.sh
```
