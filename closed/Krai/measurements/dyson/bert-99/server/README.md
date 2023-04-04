# MLPerf Inference - Bert Squad - QAIC

To run the experiments you need the following commands

## Benchmarking bert_99 model in Performance mode
```
axs byquery loadgen_output,bert_squad,qaic,kilt,framework=kilt,loadgen_scenario=Server,loadgen_mode=PerformanceOnly,model_name=bert_99,loadgen_dataset_size=10833,loadgen_buffer_size=10833,loadgen_target_qps=3430
```

