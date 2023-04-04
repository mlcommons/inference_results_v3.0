# MLPerf Inference - Bert Squad - ONNX

To run the experiments you need the following commands

## Benchmarking obert_large_95sparse_qat model in Performance mode
```
axs byquery loadgen_output,bert_squad,framework=onnx,loadgen_scenario=SingleStream,loadgen_mode=PerformanceOnly,model_name=obert_large_95sparse_qat,loadgen_dataset_size=10833,loadgen_buffer_size=10833,loadgen_target_latency=120
```

