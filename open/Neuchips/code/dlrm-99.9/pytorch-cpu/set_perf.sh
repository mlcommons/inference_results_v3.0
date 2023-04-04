
sudo cpupower frequency-set -g performance
sudo cpupower -c 0-191 idle-set -d 2
sudo echo 0 > /proc/sys/kernel/numa_balancing

#sudo cpupower frequency-set -g performance
