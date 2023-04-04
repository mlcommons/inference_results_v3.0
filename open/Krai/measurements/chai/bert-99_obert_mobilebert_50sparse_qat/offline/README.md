# MLPerf Inference - Bert Squad - ONNX

To run the experiments you need the following commands

## Benchmarking obert_mobilebert_50sparse_qat model in Performance mode
```
axs byquery loadgen_output,bert_squad,framework=onnx,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,model_name=obert_mobilebert_50sparse_qat,loadgen_dataset_size=10833,loadgen_buffer_size=10833,loadgen_target_qps=35
```

