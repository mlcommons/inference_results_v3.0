
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo 0 | sudo tee /proc/sys/kernel/numa_balancing
echo 100 | sudo tee /sys/devices/system/cpu/intel_pstate/min_perf_pct


# Clean resources
echo never  | sudo tee /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
echo never  | sudo tee /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
echo 1 | sudo tee /proc/sys/vm/compact_memory; sleep 1
echo 3 | sudo tee /proc/sys/vm/drop_caches; sleep 1
