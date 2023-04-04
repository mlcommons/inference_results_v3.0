# MLPerf Inference - Image Classification - TFLite

## SingleStream

- Set up [`program:image-classification-tflite-loadgen`](https://github.com/krai/ck-mlperf/blob/master/program/image-classification-tflite-loadgen/README.md) on your SUT.
- Customize the examples below for your SUT.

### Workloads

- [ResNet50](#resnet50)
- [EfficientNet](#efficientnet)
- [MobileNet-v1](#mobilenet_v1)
- [MobileNet-v2](#mobilenet_v2)
- [MobileNet-v3](#mobilenet_v3)

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

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --verbose \
--group.closed --scenario=singlestream --dataset_size=50000 \
--model=resnet50 --library=tflite-v2.7.1-ruy --target_latency=500 --sut=odroid
```

#### Accuracy

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model=resnet50 --mode=accuracy --scenario=singlestream --dataset_size=50000 \
--verbose --sut=odroid
```

#### Performance

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model=resnet50 --mode=performance --scenario=singlestream --target_latency=500 \
--verbose --sut=odroid
```

#### Compliance

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model=resnet50 --scenario=singlestream --compliance,=TEST04,TEST05,TEST01 \
--verbose --sut=odroid
```

<a name="efficientnet"></a>
### EfficientNet

#### "All-in-one"

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --group.open --dataset_size=50000 \
--target_latency_file=$(ck find program:image-classification-tflite-loadgen)/target_latency.txt --sut=odroid
```

#### Accuracy

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=accuracy --scenario=singlestream --dataset_size=50000 \
--verbose --sut=odroid
```

#### Performance

##### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream --target_latency=20 \
--verbose --sut=odroid
```

##### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=range_singlestream --max_query_count=256 \
--verbose --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.tflite,inference_engine_version.v2.7.1 | \
sort | tee $(ck find program:image-classification-tflite-loadgen)/target_latency.txt
```

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-tflite-loadgen)/target_latency.txt \
--verbose --sut=odroid
```

<a name="mobilenet_v1"></a>
### MobileNet-v1

#### "All-in-one"

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --group.open --dataset_size=50000 \
--target_latency_file=$(ck find program:image-classification-tflite-loadgen)/target_latency.txt --sut=odroid
```

#### Accuracy

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=accuracy --scenario=singlestream --dataset_size=50000 \
--verbose --sut=odroid
```

#### Performance

##### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream --target_latency=5 \
--verbose --sut=odroid
```

##### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=range_singlestream --max_query_count=1024 \
--verbose --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.tflite,inference_engine_version.v2.7.1 | \
sort | tee $(ck find program:image-classification-tflite-loadgen)/target_latency.txt
```

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-tflite-loadgen)/target_latency.txt \
--verbose --sut=odroid
```

<a name="mobilenet_v2"></a>
### MobileNet-v2

#### "All-in-one"

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized --scenario=singlestream --group.open --dataset_size=50000 \
--target_latency_file=$(ck find program:image-classification-tflite-loadgen)/target_latency.txt --sut=odroid
```

#### Accuracy

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=accuracy --scenario=singlestream --dataset_size=50000 \
--verbose --sut=odroid
```

#### Performance

##### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream --target_latency=3 \
--verbose --sut=odroid
```

##### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=range_singlestream --max_query_count=1024 \
--verbose --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.tflite,inference_engine_version.v2.7.1 | \
sort | tee $(ck find program:image-classification-tflite-loadgen)/target_latency.txt
```

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:) \
--model_extra_tags,=non-quantized,quantized \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-tflite-loadgen)/target_latency.txt \
--verbose --sut=odroid
```

<a name="mobilenet_v3"></a>
### MobileNet-v3

**NB:** Note that unlike [MobileNet-v1](#mobilenet_v1), [MobileNet-v2](#mobilenet_v2) and [EfficientNet](#efficientnet), [MobileNet-v3](#mobilenet_v3) does **not** require providing `--model_extra_tags,=non-quantized,quantized`.

#### "All-in-one"

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy --verbose \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--scenario=singlestream --group.open --dataset_size=50000 \
--target_latency_file=$(ck find program:image-classification-tflite-loadgen)/target_latency.txt --sut=odroid
```

#### Accuracy

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--mode=accuracy --scenario=singlestream --dataset_size=50000 \
--verbose --sut=odroid
```

#### Performance

##### Use a uniform target latency

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--mode=performance --scenario=singlestream --target_latency=6 \
--verbose --sut=odroid
```

##### [Estimate target latencies](https://github.com/krai/ck-mlperf/tree/master/program/generate-target-latency)

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--mode=performance --scenario=range_singlestream --max_query_count=1024 \
--verbose --sut=odroid
```

```bash
$(ck find program:generate-target-latency)/run.py --tags=inference_engine.tflite,inference_engine_version.v2.7.1 | \
sort | tee $(ck find program:image-classification-tflite-loadgen)/target_latency.txt
```

```bash
time ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=tflite-v2.7.1-ruy \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:) \
--mode=performance --scenario=singlestream \
--target_latency_file=$(ck find program:image-classification-tflite-loadgen)/target_latency.txt \
--verbose --sut=odroid
```
