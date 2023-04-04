## Design and Overview
For detailed guidelines see [Loadgen-bridge-design](/closed/Intel/code/common/assets/MLPerf-Loadgen-Interface.pptx)

## Instructions For Developers
### Understanding Run.py
This file takes input from workload files created by developers. To use this developers need to add their workload profile to profiles.py which is a module imported by run.py.

```
usage: Parses global and workload-specific arguments [-h] --workload-name
                                                     WORKLOAD_NAME
                                                     [--scenario {Offline,Server}]
                                                     [--mlperf-conf MLPERF_CONF]
                                                     [--user-conf USER_CONF]
                                                     [--mode {Accuracy,Performance}]
                                                     [--num-sockets NUM_SOCKETS]
                                                     [--workload-config WORKLOAD_CONFIG]
                                                     [--perf-count PERF_COUNT]
                                                     [--batch-size BATCH_SIZE]
                                                     [--num-instance NUM_INSTANCE]
                                                     [--cpus-per-instance CPUS_PER_INSTANCE]
                                                     [--warmup WARMUP]
                                                     [--log-latencies]
                                                     [--precision {int8,bf16,fp32,mix}]
                                                     --cpus-per-process
                                                     CPUS_PER_PROCESS
                                                     [--cpus-per-socket CPUS_PER_SOCKET]
                                                     [--processes-per-socket PROCESSES_PER_SOCKET]
                                                     [--enable-profiling ENABLE_PROFILING]
Parses global and workload-specific arguments: error: the following arguments are required: --workload-name, --cpus-per-process

optional arguments:
  -h, --help            show this help message and exit
  --workload-name WORKLOAD_NAME
                        Name of workload
  --scenario {Offline,Server}
                        MLPerf scenario to run
  --mlperf-conf MLPERF_CONF
                        Path to mlperf.conf file
  --user-conf USER_CONF
                        Path to user.conf file containing overridden workload
                        params
  --mode {Accuracy,Performance}
                        MLPerf mode to run
  --cores_per_instance CORES_PER_INSTANCE
                        #cores to allocate per instance
  --num_sockets NUM_SOCKETS
                        Number of sockets on the system. Not fully implemented
                        for utilizing this as yet
  --workload-config WORKLOAD_CONFIG
                        A json file that contains Workload related arguments
                        for creating sut and dataset instances
  --perf_count PERF_COUNT
                        Performance sample count to use for Performance
  --batch-size BATCH_SIZE
                        Batch size if fixed
  --num-instance NUM_INSTANCE
                        Number of instances/consumers
  --cpus-per-instance CPUS_PER_INSTANCE
                        Number of cores per instance
  --warmup WARMUP       Number of warmup iterations
  --log-latencies       Log latencies for input queue, output queue, e2e

```

### How to add workload config

Application uses workload config provided to ```run.py``` via the ```--workload-config```  runtime parameter.

This is a json file with workload related parameters. Each key-params (e.g dataset-params) is a map of keyword-value pairs required for instantiating the respective class (e.g dataset class created by user)

To add a new config add to the ```configs``` folder for the workload.

Example  :  The json below is config for Resnet-50 workload with Tensorflow.

<img src="/closed/Intel/code/common/assets/config.png" width="650">



### Steps to add a workload

1. Create a folder with your ```<workload-name>``` in ```workloads``` directory. Add ```Dataset.py```, ```Backend.py```, ```InQueue.py``` for the workload (See guidelines [Loadgen-bridge-design](/closed/Intel/code/common/assets/MLPerf-Loadgen-Bridge-Design.pdf) )

2. Create your workload config. Add path to the folder containing ```Dataset.py```, ```Backend.py```, ```InQueue.py```  to ```"import_path: <your folder path>"``` in config.yml.

5. Verify if it works by running ```run.py``` with global parameters.

Once the run is completed, the application closes. At this time, logs are saved in ```output_logs``` folder. The E2E latencies, inference latencies, input queue latencies, output queue would be available in ```latencies.txt```.

**NOTE**: ```latencies.txt``` helps developers to check latencies of their inqueue implementation. Implementation of inqueue class should ensure that E2E vs inference overhead is reasonable (e.g 1% of inference latency, or 1% of latency limit of server scenario)


**NOTE**: To verify accuracy of your workload, run your command with ```--mode Accuracy```. After completion, you can use workload specific accuracy evaluation tool on ```output_logs/mlperf_log_accuracy.json``` to evaluate accuracy.

## Code Diagram
<img src="/closed/Intel/code/common/assets/design.PNG" width="650">
