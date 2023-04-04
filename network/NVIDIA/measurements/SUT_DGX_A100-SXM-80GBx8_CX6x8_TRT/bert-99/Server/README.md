To run this benchmark, first follow the setup steps in `network/NVIDIA/README.md`. Then to generate the TensorRT engines and run the harness from LON node:

```
make generate_engines RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=lon_node"
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --test_mode=AccuracyOnly --config_ver=lon_node"
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --test_mode=PerformanceOnly --config_ver=lon_node"
```

For more details, please refer to `network/NVIDIA/README.md`.