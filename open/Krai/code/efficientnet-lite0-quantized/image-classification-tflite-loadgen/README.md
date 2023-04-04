# MLPerf Inference - Image Classification - TFLite

This C++ implementation runs TFLite models for Image Classification using TFLite.

Follow the detailed installation and benchmarking instructions in this Jupyter [notebook](https://github.com/krai/ck-mlperf/tree/master/jnotebook/image-classification-tflite-loadgen).

## Prerequisites

### [Preprocess ImageNet on an x86 machine](https://github.com/arm-software/armnn-mlperf#preprocess-on-an-x86-machine-and-detect-on-an-arm-dev-board)

#### `model-tflite-mlperf-resnet*`, `model-tflite-mlperf-efficientnet-lite0`, `model-tf-and-tflite-mlperf-mobilenet*` (resolution 224)

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.224,full --ask
```

#### `model-tf-and-tflite-mlperf-mobilenet*` (resolution 192)

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.192,full --ask
```

#### `model-tf-and-tflite-mlperf-mobilenet*` (resolution 160)

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.160,full --ask
```

#### `model-tf-and-tflite-mlperf-mobilenet*` (resolution 128)

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.128,full --ask
```

#### `model-tf-and-tflite-mlperf-mobilenet*` (resolution 96)

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.96,full --ask
```

#### `model-tflite-mlperf-efficientnet-lite1`

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.240,full --ask
```

#### `model-tflite-mlperf-efficientnet-lite2`

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.260,full --ask
```

#### `model-tflite-mlperf-efficientnet-lite3`

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.280,full --ask
```

#### `model-tflite-mlperf-efficientnet-lite4`

```bash
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv,side.300,full --ask
```

### [Detect ImageNet on a dev board](https://github.com/arm-software/armnn-mlperf#preprocess-on-an-x86-machine-and-detect-on-an-arm-dev-board)

Copy a preprocessed ImageNet dataset onto a dev board e.g. under `/datasets` and register it with CK according to its resolution e.g.:

```bash
$ echo opencv-side.240 | ck detect soft --tags=dataset,imagenet,preprocessed,rgb8 \
--extra_tags=using-opencv,crop.875,full,inter.linear,side.240 \
--full_path=/datasets/dataset-imagenet-preprocessed-using-opencv-crop.875-full-inter.linear-side.240/ILSVRC2012_val_00000001.rgb8
```

## Benchmark performance via the "classical" CK interface

### Single Stream

#### Performance

```bash
firefly$ ck benchmark program:image-classification-tflite-loadgen \
--speed --repetitions=1 --skip_print_timers \
--env.CK_VERBOSE=1 \
--env.CK_LOADGEN_SCENARIO=SingleStream \
--env.CK_LOADGEN_MODE=PerformanceOnly \
--env.CK_LOADGEN_DATASET_SIZE=1024 \
--env.CK_LOADGEN_BUFFER_SIZE=1024 \
--dep_add_tags.library=tflite,v2.7 \
--dep_add_tags.images=side.224,preprocessed \
--dep_add_tags.weights=tflite,resnet \
--dep_add_tags.compiler=gcc,v11 \
--dep_add_tags.python=v3.7
...
```

## Benchmark via the "neoclassical" CK interface ([`module:cmdgen`](https://github.com/krai/ck-mlperf/tree/master/module/cmdgen))

### [Single Stream](https://github.com/krai/ck-mlperf/blob/master/program/image-classification-tflite-loadgen/README.singlestream.md)

#### Performance

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.0-ruy \
--scenario=singlestream --mode=performance --model=resnet50 --target_latency=70 \
--verbose --sut=xavier
```
