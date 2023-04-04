# Instructions

### Quantization

```sh
# Resnet50
python code/scripts/calibration/resnet/quantization.py \
        --data_path /PATH/TO/IMAGENET_DIR \
        --cal_image_list_option /PATH/TO/cal_image_list_option_2.txt \
        --pretrained_weight /PATH/TO/PRETRAINED_RESNET50.pth \
        --pt /PATH/TO/OUTPUT_GRAPH.pt
```

### Prepare dataset & model

```sh
python code/scripts/prepare_dataset.py imagenet --dataset-path /DATASET/PATH
python code/scripts/prepare_model.py resnet50 SingleStream
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
./build/rebel_mlperf vision/classification SingleStream [PerformanceOnly/AccuracyOnly]
```
