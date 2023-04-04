# Neural Magic ResNet Implementation

The integration of [DeepSparse](https://github.com/neuralmagic/deepsparse) with the [mlcommons/inference](https://github.com/mlcommons/inference) reference implementation can be found at [neuralmagic/inference](https://github.com/neuralmagic/inference/tree/deepsparse/vision/classification_and_detection/python).

This integration of [DeepSparse](https://github.com/neuralmagic/deepsparse) with the reference implementation can be executed through the [MLCommons Collective Mind framework (CK2)](https://github.com/mlcommons/ck) through a command like this:

## ResNet50 Offline
```
cm run script --tags=run,mlperf,inference,generate-run-cmds,_all-modes,_submission,_full  \
   --adr.python.name=mlperf \
   --adr.python.version_min=3.8 \
   --adr.compiler.tags=gcc \
   --submitter=NeuralMagic \
   --implementation=reference \
   --compliance=no \
   --model=resnet50 \
   --precision=int8 \
   --backend=deepsparse \
   --device=cpu \
   --scenario=Offline \
   --mode=performance \
   --execution_mode=valid \
   --adr.imagenet-preprocessed.tags=_pytorch \
   --adr.mlperf-inference-implementation.dataset=imagenet_pytorch \
   --adr.mlperf-inference-implementation.model=zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned85_quant-none-vnni \
   --adr.mlperf-inference-implementation.max_batchsize=16 \
   --adr.mlperf-inference-implementation.num_threads=48 \
   --env.DEEPSPARSE_NUM_STREAMS=24 \
   --env.ENQUEUE_NUM_THREADS=2 \
   --target_qps=20480 \
   --offline_target_qps=20480
```

ONNXRuntime can be accessed in a similar way through setting `--backend=onnxruntime` with default settings.