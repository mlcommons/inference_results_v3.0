from argparse import ArgumentParser
import os
import sys
import time
import multiprocessing as mp
import array
import numpy as np
import threading
import subprocess
import logging
import collections

#from profiles import *
import mlperf_loadgen as lg

sys.path.insert(0, os.path.join(os.getcwd(),"common"))
from configParser import parseWorkloadConfig

#import intel_pytorch_extension as ipex
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("INTEL-Inference")


MS=1000
SCENARIO_MAP = {
    "singlestream": lg.TestScenario.SingleStream,
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}

def get_args():
    
    parser = ArgumentParser("Parses global and workload-specific arguments")
    parser.add_argument("--workload-name", help="Name of workload", required=True)
    parser.add_argument("--scenario", choices=["Offline", "Server"], help="MLPerf scenario to run", default="Offline")
    parser.add_argument("--mlperf-conf", help="Path to mlperf.conf file")
    parser.add_argument("--user-conf", help="Path to user.conf file containing overridden workload params")
    parser.add_argument("--mode", choices=["Accuracy", "Performance"], help="MLPerf mode to run", default="Performance")
    parser.add_argument("--workload-config", help="A json file that contains Workload related arguments for creating sut and dataset instances")
    parser.add_argument("--num-instance", type=int, help="Number of instances/consumers", default=2)
    parser.add_argument("--cpus-per-instance", type=int, help="Number of cores per instance", default=8)
    parser.add_argument("--warmup", type=int, help="Number of warmup iterations", default=10)
    parser.add_argument("--precision", choices=["int8", "bf16", "fp32", "mix"], help="Model precision to run", default="int8")
    parser.add_argument("--workers-per-instance", type=int, help="Number of workers per each instance/consumer", default = 1)
    parser.add_argument("--cores-offset", type=int, help="Cpus to offset on 1st socket", default=0)
    args = parser.parse_args()
    return args


class Consumer(mp.Process):
    def __init__(self, task_queue, out_queue, sut_params, dataset_params, lock, init_counter, proc_idx,
                       start_core_idx, num_cores, args, num_workers=1):
        mp.Process.__init__(self)
        self.num_workers = num_workers
        self.task_queue = task_queue
        self.out_queue = out_queue
        self.lock = lock
        self.init_counter = init_counter
        self.proc_idx = proc_idx
        self.args = args
        self.affinity = list(range(start_core_idx, start_core_idx + num_cores))
        self.start_core_idx = start_core_idx
        self.end_core_idx = start_core_idx + num_cores - 1
        self.dataset_params = dataset_params

        self.num_cores = num_cores
        self.cpus_per_worker = num_cores // num_workers
        self.workers = []
        self.sut_params = sut_params
        self.out_queue = out_queue
        self.warmup_count = args.warmup
        self.latencies = collections.defaultdict(list)
        log.info("Cores: {}-{}".format(self.start_core_idx, self.end_core_idx))


    def do_warmup(self):
        warmup_data = self.data_obj.get_warmup_samples()
        log.info("Starting warmup with {} samples".format(self.warmup_count))
        for idx in range(self.warmup_count):
            output = self.sut_obj.predict(warmup_data.data)
        log.info("Warmup Completed")


    def handle_tasks(self, i, task_queue, result_queue, args, pid, start_core, end_core):

        pid = os.getpid()
        worker_name = str(pid) + "-" + str(i)

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        #TODO: Enable profiling here
        while True:
            next_task = task_queue.get()
            if next_task is None:
                log.info("{} : Exiting ".format(worker_name))
                break

            query_id_list = next_task.query_id_list
            sample_index_list = next_task.sample_index_list

            data = self.data_obj.get_samples(sample_index_list)

            output = self.sut_obj.predict(data.data)

            result = self.data_obj.post_process(query_id_list, sample_index_list, output)

            result_queue.put(result)
            task_queue.task_done()


    def run(self):
        self.proc_idx = self.pid
        os.environ['OMP_NUM_THREADS'] = '{}'.format(self.end_core_idx-self.start_core_idx+1)
        os.sched_setaffinity(self.proc_idx, self.affinity)

        # Load model
        log.info("Loading model")
        from Backend import Backend
        self.sut_obj = Backend(**self.sut_params)
        model = self.sut_obj.load_model()

        # Load dataset (if not loaded already)
        from Dataset import Dataset
        self.data_obj = Dataset(**self.dataset_params)

        self.data_obj.load_dataset()
        log.info("Available samples: {} ".format(self.data_obj.count))


        if (self.warmup_count>0):
            self.do_warmup()

        cur_step = 0
        log.info("Start testing...")

        if self.num_workers > 1 :
            start_core = self.start_core_idx
            cores_left = self.num_cores

            for i in range(self.num_workers):
                end_core = start_core + self.cpus_per_worker - 1
                cores_left -= self.cpus_per_worker
                
                #TODO: Move this to workload config. Remove hardcoded constraints
                if cores_left < 2:
                    end_cores = self.end_core_idx
                worker = mp.Process(target=self.handle_tasks, args=(i, self.task_queue, self.out_queue, self.args, self.pid, start_core, end_core))

                self.workers.append(worker)
                worker.start()
                start_core += self.cpus_per_worker

                if cores_left < 2:
                    break

            for w in self.workers:
                w.join()
            log.info("{} : Exiting consumer process".format(os.getpid()))
        else:
            self.handle_tasks(0, self.task_queue, self.out_queue, self.args, self.pid, self.start_core_idx, self.end_core_idx)


def response_loadgen(out_queue):

    while True:
        next_task = out_queue.get()
        if next_task is None:
            # None means shutdown
            log.info('Exiting response thread')
            break
        query_id_list = next_task.query_id_list
        result = next_task.result
        array_type_code = next_task.array_type_code

        batch_size = len(query_id_list)

        for id, out in zip(query_id_list, result):
            response_array = array.array(array_type_code, out)
            bi = response_array.buffer_info()
            responses = [lg.QuerySampleResponse(id, bi[0], bi[1]*response_array.itemsize)]
            lg.QuerySamplesComplete(responses)

def flush_queries():
    pass

#def process_latencies(latencies):
#    pass

def load_query_samples(query_samples):
    pass

def unload_query_samples(query_samples):
    pass

def main():
    args = get_args()

    # Get workload config parameters
    backend_params, dataset_params, enqueue_params, buckets, num_resp_qs, import_path = parseWorkloadConfig(args.workload_config)

    sys.path.insert(0, os.path.join(os.getcwd(),import_path))

    # Imports
    from InQueue import InQueue

    global num_cpus
    
    log.info(args)
    scenario = args.scenario
    mode = args.mode

    # TODO: Need to validate the cpu-instance combo is valid on the system
    num_ins = args.num_instance
    num_cpus = args.cpus_per_instance
    ins_per_consumer = args.workers_per_instance

    # Establish communication queues
    lock = mp.Lock()
    init_counter = mp.Value("i", 0)
    manager = mp.Manager()

    settings = lg.TestSettings()
    settings.scenario = SCENARIO_MAP[scenario.lower()]
    settings.FromConfig(args.mlperf_conf, args.workload_name, scenario)
    settings.FromConfig(args.user_conf, args.workload_name, scenario)
    settings.mode = lg.TestMode.AccuracyOnly if mode.lower()=="accuracy" else lg.TestMode.PerformanceOnly

    consumers = []
    loadgen_cores = args.cores_offset

    # TODO: Assign response queues to instances based on socket/numa config
    out_queue = [mp.Queue() for _ in range(num_resp_qs)]
    if len(buckets)==0:
        # If no 'bucketing', all consumers/instances fetch from a single input queue
        input_queues = mp.JoinableQueue()

        total_ins = num_ins

        i = 0
        # TODO: Consider system config i.e core-socket-numa allocation
        while i < num_ins:
            start_core_idx = i * num_cpus + loadgen_cores
            consumer = Consumer(input_queues, out_queue[i%num_resp_qs], backend_params, dataset_params, lock, init_counter, i, start_core_idx, num_cpus, args, ins_per_consumer)
            consumers.append(consumer)
            i += 1
    else:
        batch_sizes = buckets["batch_sizes"]
        cutoffs = buckets["cutoffs"]
        bucket_instances = buckets["instances"]
        cpus_per_instances = buckets["cores_per_bucket_instances"]
        ins_per_bucket_consumers = buckets.get("instances_per_bucket_consumers",[1]*len(cutoffs))

        input_queues = [mp.JoinableQueue() for _ in range(len(cutoffs))]
        total_ins = 0
        i = 0
        start_core_idx = 0 + loadgen_cores
        for j, cutoff in enumerate(cutoffs):
            batch_size = batch_sizes[j]
            num_cpus = cpus_per_instances[j]
            num_ins = bucket_instances[j]
            ins_per_consumer = ins_per_bucket_consumers[j]
            total_ins += num_ins
            
            for _ in range(num_ins):
                # TODO: Need to align with NUMA configuration 
                #       so that each consumer's cores are on same NUMA node
                log.info("Assigning to output queue {}".format( i % num_resp_qs))
                consumer = Consumer(input_queues[j], out_queue[i%num_resp_qs], backend_params, dataset_params, lock, init_counter, i, start_core_idx, num_cpus, args, ins_per_consumer)
                consumers.append(consumer)
                
                start_core_idx += num_cpus
                i += 1
    
    # Update enqueue parameters and instantiate object        
    enqueue_params.update(mpQueue=input_queues, **buckets) #, qsl=datasetObj)
    enqueueObj = InQueue(**enqueue_params)

    for c in consumers:
        c.start()

    # Wait until all sub-processors are ready
    while init_counter.value < total_ins:
        time.sleep(2)

    # Start response thread(s)
    resp_workers = [threading.Thread(
        target=response_loadgen, args=(out_queue[i],)) for i in range(num_resp_qs)]

    for resp_worker in resp_workers:
        resp_worker.daemon = True
        resp_worker.start()

    def issue_queries(query_samples):
        enqueueObj.put(query_samples, receipt_time=time.time())


    sut = lg.ConstructSUT(
#        issue_queries, flush_queries, process_latencies)
        issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        dataset_params['total_sample_count'], min(dataset_params['total_sample_count'], settings.performance_sample_count_override), load_query_samples, unload_query_samples)

    log_path = "output_logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    log.info("Test completed")

    if len(buckets) > 0:
        for q in input_queues:
            for i in range(bucket_instances[j]):
                q.put(None)
    else:
        for _ in range(init_counter.value):
            input_queues.put(None)

    for c in consumers:
        c.join()
    
    for out_q in out_queue:
        out_q.put(None)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


    
if __name__ == "__main__":
    main()
