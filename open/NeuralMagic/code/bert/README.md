# Neural Magic BERT Implementation

The integration of [DeepSparse](https://github.com/neuralmagic/deepsparse) with the [mlcommons/inference](https://github.com/mlcommons/inference) reference implementation can be found at [neuralmagic/inference](https://github.com/neuralmagic/inference/tree/deepsparse/language/bert).

This integration of [DeepSparse](https://github.com/neuralmagic/deepsparse) with the reference implementation can be executed through the [MLCommons Collective Mind framework (CK2)](https://github.com/mlcommons/ck) through a command like this:

### BERT-99%: oBERT-Large Offline
```
cm run script --tags=run,mlperf,inference,generate-run-cmds,_submission,_all-modes,_full  \
   --adr.python.name=mlperf \
   --adr.python.version_min=3.8 \
   --adr.compiler.tags=gcc \
   --submitter=NeuralMagic \
   --implementation=reference \
   --compliance=no \
   --model=bert-99 \
   --precision=int8 \
   --backend=deepsparse \
   --device=cpu \
   --scenario=Offline \
   --mode=performance \
   --execution_mode=valid \
   --adr.mlperf-inference-implementation.max_batchsize=384 --target_qps=1280 --offline_target_qps=1280 \
   --adr.mlperf-inference-implementation.model=zoo:nlp/question_answering/obert-large/pytorch/huggingface/squad/pruned95_quant-none-vnni
```

ONNXRuntime can be accessed in a similar way through setting `--backend=onnxruntime` with default settings.