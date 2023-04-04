To run this benchmark, first follow the setup steps in `closed/NVIDIA/README_Triton_CPU.md`. Then to run the harness:

```
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --test_mode=PerformanceOnly"
```

For more details, please refer to `closed/NVIDIA/README_Triton_CPU.md`.