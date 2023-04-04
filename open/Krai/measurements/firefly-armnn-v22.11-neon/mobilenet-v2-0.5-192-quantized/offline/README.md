# MLPerf Inference - Image Classification - ArmNN-TFLite

## Single Stream

- Set up [`program:image-classification-armnn-tflite-loadgen`](https://github.com/krai/ck-mlperf/blob/master/program/image-classification-armnn-tflite-loadgen/README.md) on your SUT.
- Customize the examples below for your SUT. In particular, adjust your `--target_latency` according to your `--sut` and `--library`.

### Workloads

- [ResNet50](#resnet50)
- [EfficientNet](#efficientnet)
- [MobileNet-v1](#mobilenet_v1)
- [MobileNet-v2](#mobilenet_v2)
- [MobileNet-v3](#mobilenet_v3)

<a name="resnet50"></a>
### ResNet50

The following table gives some measured `--target_latency` values.

| `--sut`     | `--library` (+version+backend) | `--target_latency` (ms) | Round                                              | Notes                            |
|-------------|--------------------------------|-------------------------|----------------------------------------------------|----------------------------------|
| `firefly`   | `armnn-v22.11-neon`            | 128                     | v2.x                                               | RK3588S (RK3399 previously).     |
|             | `armnn-v22.11-opencl`          | 82                      | v2.x                                               |                                  |
| `vim4`      | `armnn-v22.11-neon`            | 195                     | v2.x                                               |                                  |
|             | `armnn-v22.11-opencl`          | 148                     | v2.x                                               |                                  |
| `xavier`    | `armnn-v21.11-neon`            | 66                      | [v2.0](https://mlcommons.org/en/inference-edge-20) | Observed 58 ms (up to -20% vs v0.5-v1.1).  |
| `odroid`    | `armnn-v21.11-neon`            | 339                     | [v2.0](https://mlcommons.org/en/inference-edge-20) | +10% vs `rpi4` [despite](https://github.com/ARM-software/armnn/issues/600) 2.2 GHz vs 1.5 GHz? |
|             | `armnn-v21.11-opencl`          | 246                     | [v2.0](https://mlcommons.org/en/inference-edge-20) | Mali-G52 MP2 is faster than CPU. |
| `rpi4`      | `armnn-v21.11-neon`            | 349                     | [v2.0](https://mlcommons.org/en/inference-edge-20) | Fan: on.                         |
| `rpi3`      | `armnn-v21.11-neon`            | 2311                    | [v2.0](https://mlcommons.org/en/inference-edge-20) |                                  |
| `rpi4`      | `armnn-v21.05-neon`            | 312                     | [v1.1](https://mlcommons.org/en/inference-edge-11) | Fan: on.                         |
|             |                                | 313                     | [v1.1](https://mlcommons.org/en/inference-edge-11) | Fan: off.                        |
| `xavier`    | `armnn-v21.05-neon`            | 72                      | [v1.1](https://mlcommons.org/en/inference-edge-11) | Power mode: MAXN.                |
|             |                                | 378                     | [v1.1](https://mlcommons.org/en/inference-edge-11) | Power mode: MODE_15W.            |
|             |                                | 252                     | [v1.1](https://mlcommons.org/en/inference-edge-11) | Power mode: MODE_30W_ALL.        |
|             |                                | 259                     | [v1.1](https://mlcommons.org/en/inference-edge-11) | Power mode: MODE_30W_6CORE.      |
|             |                                | 213                     | [v1.1](https://mlcommons.org/en/inference-edge-11) | Power mode: MODE_30W_4CORE.      |
| `xavier`    | `armnn-v21.02-neon`            | 69                      | [v1.0](https://mlcommons.org/en/inference-edge-10) | -4% vs v0.7. (10 min vs 1 min duration?) |
| `firefly`   | `armnn-v21.02-neon`            | 415                     | [v1.0](https://mlcommons.org/en/inference-edge-10) | +6% vs v0.5.                     |
|             | `armnn-v21.02-opencl`          | 556                     | [v1.0](https://mlcommons.org/en/inference-edge-10) | +24 vs v0.5.                     |
| `firefly`   | `armnn-v20.08-neon`            | 368                     | [v0.7](https://mlcommons.org/en/inference-edge-07) | -6% vs v0.5.                     |
|             | `armnn-v20.08-opencl`          | 458                     | [v0.7](https://mlcommons.org/en/inference-edge-07) | +2% vs v0.5.                     |
| `xavier`    | `armnn-v20.08-neon`            | 73                      | [v0.7](https://mlcommons.org/en/inference-edge-07) | Power mode: MAXN.                |
| `rpi4`      | `armnn-v20.08-neon`            | 464                     | [v0.7](https://mlcommons.org/en/inference-edge-07) | 32-bit Linux (Debian 11).        |
| `rpi4`      | `armnn-v20.08-neon`            | 319                     | [v0.7](https://mlcommons.org/en/inference-edge-07) | 64-bit Linux (Ubuntu 20.04).     |
| `rpi4`      | `armnn-v19.08-neon`            | 448                     | [v0.5](https://mlcommons.org/en/inference-edge-05) |                                  |
| `firefly`   | `armnn-v19.08-neon`            | 391                     | [v0.5](https://mlcommons.org/en/inference-edge-05) |                                  |
|             | `armnn-v19.08-opencl`          | 448                     | [v0.5](https://mlcommons.org/en/inference-edge-05) | Mali-T860 MP4 slower than CPU.   |
| `mate10pro` | `armnn-v19.08-neon`            | 495                     | [v0.5](https://mlcommons.org/en/inference-edge-05) |                                  |
|             | `armnn-v19.08-opencl`          | 354                     | [v0.5](https://mlcommons.org/en/inference-edge-05) | Mali-G72 MP12 faster than CPU.   |
| `hikey960`  | `armnn-v19.08-neon`            | 495                     | [v0.5](https://mlcommons.org/en/inference-edge-05) |                                  |
|             | `armnn-v19.08-opencl`          | 204                     | [v0.5](https://mlcommons.org/en/inference-edge-05) | Mali-G71 MP8 faster than CPU... and Mali-G72 MP12! |

#### "All-in-one"

Specifying `--group.closed` runs the benchmark in the following modes required for the Closed division:
- Accuracy with the given `--dataset_size`.
- Performance with the given `--target_latency`.
- Compliance tests (TEST01, TEST04, TEST05) with the given `--target_latency`.

Specifying `--group.open` runs the benchmark in the following modes required for the Open division:
- Accuracy with the given `--dataset_size`.
- Performance with the given `--target_latency`.


**NB:** This mode is supported only with CK &leq; v1.17.0 or &geq; v2.6.0:

```
python3 -m pip install ck==2.6.1
```
<a name="resnet50"></a>
### ResNet50

#### "All-in-one"

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--group.closed --scenario=singlestream --dataset_size=50000 \
--model=resnet50 --library=armnn-v22.11-neon --target_latency=350 --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--group.closed --scenario=singlestream --dataset_size=50000 \
--model=resnet50 --library=armnn-v22.11-opencl --target_latency=250 --sut=odroid
```

#### Accuracy

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--model=resnet50 --library=armnn-v22.11-neon --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--model=resnet50 --library=armnn-v22.11-opencl --sut=odroid
```

#### Performance

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--scenario=singlestream --mode=performance \
--model=resnet50 --library=armnn-v22.11-neon --target_latency=350 --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--scenario=singlestream --mode=performance \
--model=resnet50 --library=armnn-v22.11-opencl --target_latency=250 --sut=odroid
```

#### Compliance

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--scenario=singlestream --compliance,=TEST04,TEST05,TEST01 \
--model=resnet50 --library=armnn-v22.11-neon --target_latency=350 --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--scenario=singlestream --compliance,=TEST04,TEST05,TEST01 \
--library=armnn-v22.11-opencl --model=resnet50 --target_latency=250 --sut=odroid
```

<a name="efficientnet"></a>
### EfficientNet

#### "All-in-one"

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --group.open --dataset_size=50000 \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --group.open --dataset_size=50000 \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

#### Accuracy

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=accuracy --dataset_size=50000 --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=accuracy --dataset_size=50000 --sut=odroid
```

#### Performance

##### Neon

###### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=performance --target_latency=10 --sut=odroid
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=range_singlestream --max_query_count=256 --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v22.11 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt
```
```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

##### OpenCL

###### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=performance --target_latency=10 --sut=odroid
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=range_singlestream --max_query_count=256 --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v22.11 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt
```

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

<a name="mobilenet_v1"></a>
### MobileNet-v1

#### "All-in-one"

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --group.open --dataset_size=50000 \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --group.open --dataset_size=50000 \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

#### Accuracy

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=accuracy --dataset_size=50000 --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=accuracy --dataset_size=50000 --sut=odroid
```

#### Performance

##### Neon

###### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=performance --target_latency=5 --sut=odroid
```

##### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=range_singlestream --max_query_count=256 --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v22.11 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt
```

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

##### OpenCL

###### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=performance --target_latency=5 --sut=odroid
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=range_singlestream --max_query_count=256 --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v22.11 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt
```

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

<a name="mobilenet_v2"></a>
### MobileNet-v2

#### "All-in-one"

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --group.open --dataset_size=50000 \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --group.open --dataset_size=50000 \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

#### Accuracy

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=accuracy --dataset_size=50000 --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=accuracy --dataset_size=50000 --sut=odroid
```

#### Performance

##### Neon

###### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=performance --target_latency=5 --sut=odroid
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=range_singlestream --max_query_count=256 --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v22.11 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt
```

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-neon --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

##### OpenCL

###### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --mode=performance --target_latency=5 --sut=odroid
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=range_singlestream --max_query_count=256 --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v22.11 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt
```

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v22.11-opencl --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized --mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

<a name="mobilenet_v3"></a>
### MobileNet-v3

**NB:** Note that unlike [MobileNet-v1](#mobilenet_v1), [MobileNet-v2](#mobilenet_v2) and [EfficientNet](#efficientnet), [MobileNet-v3](#mobilenet_v3) does **not** require providing `--model_extra_tags,=non-quantized,quantized`.

#### "All-in-one"

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.odroid.txt \
--library=armnn-v22.11-neon --scenario=singlestream --group.open --dataset_size=50000 --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.odroid.txt \
--library=armnn-v22.11-opencl --scenario=singlestream --group.open --dataset_size=50000 --sut=odroid
```

#### Accuracy

##### Neon

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--scenario=singlestream --mode=accuracy --dataset_size=50000 --library=armnn-v22.11-neon --sut=odroid
```

##### OpenCL

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--scenario=singlestream --mode=accuracy --dataset_size=50000 --library=armnn-v22.11-opencl --sut=odroid
```

#### Performance

##### Neon

###### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--scenario=singlestream --mode=performance --library=armnn-v22.11-neon --target_latency=6 --sut=odroid
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--mode=performance --scenario=range_singlestream --max_query_count=256 --library=armnn-v22.11-neon --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v22.11 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt
```

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--mode=performance --scenario=singlestream  --library=armnn-v22.11-neon \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```

##### OpenCL

###### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--scenario=singlestream --mode=performance --library=armnn-v22.11-opencl --target_latency=6 --sut=odroid
```

###### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--mode=performance --scenario=range_singlestream --max_query_count=256 --library=armnn-v22.11-opencl --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.armnn,inference_engine_version.v22.11 | \
sort | tee $(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt
```

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--mode=performance --scenario=singlestream  --library=armnn-v22.11-opencl \
--target_latency_file=$(ck find program:image-classification-armnn-tflite-loadgen)/target_latency.txt --sut=odroid
```
