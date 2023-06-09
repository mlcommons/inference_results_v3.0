
# ResNet50

## ResNet50 Offline - DeepSparse

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

## ResNet50 Offline - ONNXRuntime

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
   --backend=onnxruntime \
   --device=cpu \
   --scenario=Offline \
   --mode=performance \
   --execution_mode=valid \
   --target_qps=1536 \
   --offline_target_qps=1536
```

# BERT-Large

## BERT-99%: oBERT-Large Offline - DeepSparse
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

## BERT-99%: MobileBERT Offline

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
   --adr.mlperf-inference-implementation.max_batchsize=384 \
   --target_qps=5120 --offline_target_qps=5120 \
   --adr.mlperf-inference-implementation.model=zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/14layer_pruned50_quant-none-vnni --env.DEEPSPARSE_SEQLENS="64,128,192,256,384"
```

## BERT-99.9%: MobileBERT Offline - DeepSparse

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
   --adr.mlperf-inference-implementation.max_batchsize=384 --target_qps=3072 --offline_target_qps=3072 \
   --adr.mlperf-inference-implementation.model=zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base_quant-none --env.DEEPSPARSE_SEQLENS="64,128,192,256,384"
```


## BERT-99%: BERT-Large Offline - ONNXRuntime

```
cm run script --tags=run,mlperf,inference,generate-run-cmds,_submission,_all-modes,_full  \
   --adr.python.name=mlperf \
   --adr.python.version_min=3.8 \
   --adr.compiler.tags=gcc \
   --submitter=NeuralMagic \
   --implementation=reference \
   --compliance=no \
   --model=bert-99 \
   --precision=fp32 \
   --backend=onnxruntime \
   --device=cpu \
   --scenario=Offline \
   --mode=performance \
   --execution_mode=valid
```