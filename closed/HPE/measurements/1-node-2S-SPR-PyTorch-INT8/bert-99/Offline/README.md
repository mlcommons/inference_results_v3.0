# Setup from Source
## use conda to setup env
```
  conda create -n bert-pt
  source activate bert-pt
```


## download dataset and model

 dataset

```
{dataset,model}mkdir -p bert/{dataset,model}
cd bert
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ./dataset/dev-v1.1.json
```
 model

```
git clone https://huggingface.co/bert-large-uncased model
 #replace  pytorch_model.bin
cd model
wget https://zenodo.org/record/4792496/files/pytorch_model.bin?download=1 -O pytorch_model.bin
cd ..
export DATA_PATH=/path/of/bert/
```

## prepare env

```
# need to close system numa_balancing
echo 0 > /proc/sys/kernel/numa_balancing
git clone <this_repo>
cp <this_repo>/closed/Intel/code/bert-99/pytorch-cpu/prepare* .
bash prepare_conda_env.sh
bash pepare_env.sh --code=<the/path/of/this/repo>
cd <this_repo>/closed/Intel/code/bert-99/pytorch-cpu/
```

## Run command

 #for accuracy mode, you should install tensorflow
 #please update the vocab.txt and dev-v1.1.json path in run.sh and run_server.sh.

```
bash run.sh    #for offline performance
bash run.sh --accuracy   #for offline accuracy
```
```
bash run_server.sh #for server performance
bash run_server.sh --accuracy    #for server accuracy
```
inference driver options:
    ("-m, --model_file", "[filename] Torch Model File")

    ("-s, --sample_file", "[filename] SQuAD Sample File")
    
    ("-t, --test_mode", "Test mode [Offline, Server]")
    
    ("-n, --inter_parallel", "[number] Instance Number")
    
    ("-j, --intra_parallel", "[number] Thread Number Per-Instance")
    
    ("-c, --mlperf_config", "[filename] Configuration File for LoadGen")
    
    ("-u, --user_config", "[filename] User Configuration for LoadGen")
    
    ("-o, --output_dir", "[filename] Test Output Directory")
    
    ("-w, --watermark", "[number] Sequence length watermark")
    
    ("-h, --hyperthreading", "[true/false] Whether system enabled hyper-threading or not")


For ICX above, subsitute: -mavx512cd -mavx512dq -mavx512bw -mavx512vl to -march=native


# Setup with docker
## prepare dataset and model
 follow the steps above

## start docker container
### option 1: build docker
```
  cd docker
  bash build_bert-99_container.sh
```
### option 2: pull docker
```
  <TBD: command to pull docker>
```
```
docker run --privileged --name intel_bert -itd --net=host --ipc=host -v </path/to/datatset/and/model>:/data/mlperf_date/bert intel/intel-optimized-pytorch:mlperf-submission-inference-2.1-bert99
docker ps -a #get container "id"
docker exec -it <id> bash
```

## convert dataset and model
```
cd bert-99/pytorch-cpu/
export DATA_PATH=/data/mlperf_date/bert
bash convert.sh
```
## Run command
follow the steps above









