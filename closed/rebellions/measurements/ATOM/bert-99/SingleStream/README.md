# Instructions

### Quantization

```sh
# Bert 
python code/scripts/calibration/bert/quantization.py \
        --model_path /PATH/TO/model.ckpt-5474 \
        --data_path /PATH/TO/squad_v1.1 \
        --pb2 /PATH/TO/OUTPUT_GRAPH.pb2
```

### Prepare dataset & model

```sh
python code/scripts/prepare_dataset.py squad --dataset-path /DATASET/PATH
python code/scripts/prepare_model.py bert_large SingleStream
```

### Compile

In your root directory:

```sh
export REBEL_COMPILER_PATH=/PATH/TO/REBEL_COMPILER
export REBEL_DRIVER_PATH=/PATH/TO/REBEL_REBEL_DRIVER
cmake -S code -B ./build -D CMAKE_PREFIX_PATH="$REBEL_COMPILER_PATH};${REBEL_DRIVER_PATH}"
cmake --build build --parallel ${nproc}
```

### Run

Assume you followed above compile example (using `./build` directory)

```sh
./build/rebel_mlperf language/bert SingleStream [PerformanceOnly/AccuracyOnly]
```
