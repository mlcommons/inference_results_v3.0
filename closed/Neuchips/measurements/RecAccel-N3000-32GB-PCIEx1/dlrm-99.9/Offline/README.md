To run this benchmark, first follow the setup steps in `closed/Neuchips/README.md`. Then to run the harness:

```
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --test_mode=AccuracyOnly" NEU_CARDS="1"
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --test_mode=PerformanceOnly" NEU_CARDS="1"
```

For more details, please refer to `closed/Neuchips/README.md`.
