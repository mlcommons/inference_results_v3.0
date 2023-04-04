Step-by-Step
============
This example used [Neural Engine](https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/backends/neural_engine) to get the MLPerf performance. It can test model `Bert Large Squad Sparse` and `MiniLM L12`.
The benchmark was evaluated using a server with two Intel(R) Xeon(R) Platinum 8480+ (Sapphire Rappids) CPUs with 56 cores or two Intel(R) Xeon(R) Platinum 8380 CPU (IceLake) with 40 cores.
| Benchmark      | F1 Score [%] | Machine info  |  Offline Throughput [samples/sec]  |
|:----------------:|:------:|:-------:|:--------:|
| Bert-Large (Sparse 90% structured) | 90.252 | IceLake | 333 |
| Bert-MiniLM L12 | 90.929 | IceLake | 2236.38 |
| Bert-MiniLM L12 | 91.0745 | Sapphire Rappids | 4237.26 |
## Prerequisite

# prepare C++ environment
GCC greater than 9.2.1 && cmake greater than 3.18
prepare intel oneapi
```
source /PATH_TO_ONEAPI/intel/oneapi/compiler/latest/env/vars.sh
```

# generate dataset and model
>>Note: Require a python with tensorflow, six, numpy
```
pip install tensorflow six numpy
cd ./datasets
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ./dev-v1.1.json
python gen_data.py
cd ../
pip install gdown 
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1n0_X3BbSlFoKluDqyOmX3Hhna9JCxZv2
```

# make neural engine library
```
git clone --branch mlperf_neuralengine_2023 --recursive https://github.com/intel/intel-extension-for-transformers.git
cp -r intel-extension-for-transformers/intel_extension_for_transformers/ ./ 
cd intel_extension_for_transformers/backends/neural_engine/
mkdir build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3) -DNE_WITH_SPARSELIB=True && make -j
cd -
bash install_third_party.sh
```

# install mlperf loadgen library
```
git clone --branch master --recursive https://github.com/mlcommons/inference.git
cd ./inference
mkdir loadgen/build/ && cd loadgen/build/
cmake .. && cmake --build .
```

# make mlperf sut example
```
mkdir build && cd build
CC=icx CXX=icpx cmake ..
make -j
```

## Run Command
1. Performance Mode
when you run minilm please also add --minilm=true for both performance and accuracy
when you run bert large please keep batch size as 4
please make sure "--model_conf" and "--model_weight" navigate to real model directory
```
mkdir mlperf_output
GLOG_minloglevel=2  INST_NUM=20 ./build/inference_sut --model_conf=./minilm_mha_ir/conf.yaml --model_weight=./minilm_mha_ir/model.bin  --sample_file=./datasets/ --output_dir=./mlperf_output --mlperf_config=./mlperf.conf --user_config=user.conf
```
2. Accuracy Mode
Require transformers in python
```
mkdir mlperf_output
GLOG_minloglevel=2  INST_NUM=20 ./build/inference_sut --model_conf=./minilm_mha_ir/conf.yaml --model_weight=./minilm_mha_ir/model.bin  --sample_file=./datasets/ --output_dir=./mlperf_output --mlperf_config=./mlperf.conf --user_config=user.conf  --accuracy=true
python accuracy-squad.py
```
