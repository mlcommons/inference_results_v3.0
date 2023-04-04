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

# Setup with docker
## prepare dataset and model
 follow the steps above

## start docker container
### option 1: build docker
```
  cd docker
  bash build_bert-99_container.sh
  docker run --privileged --name intel_bert -itd --net=host --ipc=host -v </path/to/datatset/and/model>:/data/mlperf_date/bert mlperf_inference_bert:3.0
  docker ps -a #get container "id"
  docker exec -it <id> bash
```

## convert dataset and model
```
cd /opt/workdir/code/bert-99/pytorch-cpu
export DATA_PATH=/data/mlperf_date/bert
bash convert.sh
```









