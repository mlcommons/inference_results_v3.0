echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
sudo echo 0 > /proc/sys/kernel/numa_balancing
sudo echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct
sudo cpupower frequency-set -g performance
