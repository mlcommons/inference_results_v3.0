This submission was prepared during the 
[1st open benchmarking and optimization challenge](https://access.cknowledge.org/playground/?action=challenges&name=57cbc3384d7640f9) organized by the 
[MLCommons taskforce on automation and reproducibility](https://cKnowledge.org/mlcommons-taskforce)
and the [cTuning foundation](https://cTuning.org) 
and powered by our [free and open-source CK technology](https://github.com/mlcommons/ck).

## Implementations Used

1. [MLCommons Reference Implementations](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/app-mlperf-inference-reference) for pytorch, tensorflow and onnxruntime backends
2. [Nvidia Implementation](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/reproduce-mlperf-inference-nvidia) (used for MLPerf Inference 2.1 round)
3. [MLCommons Taskforce C++ Implementation](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/app-mlperf-inference-reference) for onnxruntime backend
4. [TFLite C++ Implementation](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/app-mlperf-inference-tflite-cpp) for mobilenet and efficientnet models

## System Setup

1. [NVIDIA Jetson ORIN](https://github.com/mlcommons/ck/blob/master/cm-mlops/project/mlperf-inference-v3.0-submissions/docs/setup-nvidia-jetson-orin.md)
2. [AWS Cloud Instance](https://github.com/mlcommons/ck/blob/master/cm-mlops/project/mlperf-inference-v3.0-submissions/docs/setup-aws-instance.md)
3. [Google Cloud Platform](https://github.com/mlcommons/ck/blob/master/cm-mlops/project/mlperf-inference-v3.0-submissions/docs/setup-gcp-instance.md)

For other systems simply follow the setup given in the benchmark READMEs below.

## Benchmarks
1. [ResNet50](https://github.com/mlcommons/ck/blob/master/cm-mlops/challenge/optimize-mlperf-inference-v3.0-2023/docs/generate-resnet50-submission.md)
2. [Bert](https://github.com/mlcommons/ck/blob/master/cm-mlops/challenge/optimize-mlperf-inference-v3.0-2023/docs/generate-bert-submission.md)
3. [3d-unet](https://github.com/mlcommons/ck/blob/master/cm-mlops/challenge/optimize-mlperf-inference-v3.0-2023/docs/generate-3d-unet-submission.md)
4. [RNNT](https://github.com/mlcommons/ck/blob/master/cm-mlops/challenge/optimize-mlperf-inference-v3.0-2023/docs/generate-rnnt-submission.md)
5. [RetinaNet](https://github.com/mlcommons/ck/blob/master/cm-mlops/challenge/optimize-mlperf-inference-v3.0-2023/docs/generate-retinanet-submission.md)

## Power Measurement
Please follow [this tutorial](https://github.com/mlcommons/ck/blob/master/docs/tutorials/mlperf-inference-power-measurement.md) to setup power analyzer for a given System Under Test (SUT). For all our power submissions the power analyzer was connected via USB to a director machine and the SUT was communicating to the director via LAN. 
