# MLPerf Inference v3.0 Implementations

This is a repository of the `Alibaba Cloud Server E-Series SinianML Platform` using optimized implementations
from [MLPerf Inference Benchmark v3.0](https://github.com/mlcommons/inference/tree/master).

## Platform

### Sinian

Sinian is the Alibaba’s compiler-based heterogeneous hardware acceleration platform, targeting extreme performance for
machine learning (ML) applications. Interfacing with the upper level frameworks (e.g., Alibaba PAI, Tensorflow, MxNet
and Pytorch), Sinian enables deep co-optimizations between software and hardware to deliver high execution efficiency
for ML applications. Sinian is fully tailorable (“statically and dynamically”) for cloud computing, edge computing, and
IoT devices, making it easy to achieve performance portability between training, deploying and inferring ML models
across heterogeneous accelerators.

SinianML is the effective ML optimization framework in Sinian with high scalability and flexibility. It has the
following features:

1. Boost the performance of ML models by using Neural Architecture Search, Compression and Distillation, with little
   hand-crafted efforts;
2. Auto-tune and jointly optimize the heterogeneous system performance across system, framework and hardware libraries;

## Benchmarks

The following benchmarks are part of our submission for MLPerf Inference v3.0:

* Resnet50

## Scenarios

The above benchmarks can run in the following inference scenarios:

* Offline
* MultiStream
* SingleStream

Please refer to
the [MLPerf Inference official page](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#3-scenarios)
for explanations about the scenarios.

# Alibaba Submission

Our MLPerf Inference v3.0 implementation has the following submissions:

| Benchmark | System     | Submissions                   |
| :-------- | :-----     | :---------------------------- |
| Resnet50  | Edge       | 99% of FP32 accuracy, Offline |

The benchmark is stored in the `code/` directory which contains a `README.md` detailing the instructions on how to set
up the benchmark, including:

1. Downloading the dataset and model;
2. Running any necessary preprocessing;
3. Details on the optimizations being performed;
