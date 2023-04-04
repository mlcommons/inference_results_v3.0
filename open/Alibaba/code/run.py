import collections
import logging
import multiprocessing as mp
import os
import sys
import threading
from argparse import ArgumentParser

import array
import mlperf_loadgen as lg
import time
import yaml

import numpy as np

current_path = os.path.abspath(__file__)
work_dir = current_path.split('/')[:-1]
work_dir = '/'.join(work_dir)
os.chdir(work_dir)
sys.path.insert(0, os.path.join(os.getcwd(), "common"))


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ACS-Sinian-Inference")

MS = 1000
SCENARIO_MAP = {
    "multistream": lg.TestScenario.MultiStream,
    "singlestream": lg.TestScenario.SingleStream,
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}


def get_args():
    parser = ArgumentParser("Parses global and workload-specific arguments")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    return args


class Consumer(mp.Process):
    def __init__(self, task_queue, out_queue, lock, init_counter, proc_idx,
                 start_core_idx, model_param, system_param, dataset_param, mlperf_param):
        mp.Process.__init__(self)
        self.task_queue = task_queue
        self.out_queue = out_queue
        self.lock = lock
        self.init_counter = init_counter
        self.proc_idx = proc_idx
        self.affinity = list(range(start_core_idx, start_core_idx + system_param["core_per_instance"]))
        self.start_core_idx = start_core_idx
        self.end_core_idx = start_core_idx + system_param["core_per_instance"] - 1
        self.num_cores = system_param["core_per_instance"]
        self.workers = []
        self.out_queue = out_queue
        self.warmup_count = system_param["warmup"]
        self.latencies = collections.defaultdict(list)
        self.num_workers = system_param["num_worker_per_instance"]
        self.core_per_worker = system_param["core_per_worker"]
        self.dataset_param = dataset_param
        self.model_param = model_param
        self.mlperf_param = mlperf_param
        log.info("Cores: {}-{}".format(self.start_core_idx, self.end_core_idx))

    def do_warmup(self):
        warmup_data = self.data_obj.get_warmup_samples()

        from rknn_excute import RKNN_model_container
        rknn_model_container = RKNN_model_container(model_path=self.model_param["model_path"])
        
        log.info("Starting warmup")
        for idx in range(self.warmup_count):
            for item in warmup_data:
                output = rknn_model_container.run(item.data)[0]

        rknn_model_container.release()
        log.info("Warmup Completed")

    def handle_tasks(self, i, task_queue, result_queue, pid):

        pid = os.getpid()
        worker_name = str(pid) + "-" + str(i)
        from rknn_excute import RKNN_model_container
        rknn_model_container = RKNN_model_container(model_path=self.model_param["model_path"])
        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()
        # TODO: Enable profiling here
        while True:
            next_task = task_queue.get()
            
            if next_task is None:
                log.info("{} : Exiting ".format(worker_name))
                break
            query_id_list = next_task.query_id_list
            sample_index_list = next_task.sample_index_list
            data = self.data_obj.get_samples(sample_index_list)
            output = rknn_model_container.run(data.data)[0]
            result = self.data_obj.post_process(query_id_list, sample_index_list, output)

            result_queue.put(result)
            task_queue.task_done()

        rknn_model_container.release()

    def run(self):
        self.proc_idx = self.pid
        os.environ['OMP_NUM_THREADS'] = '{}'.format(self.num_workers)
        os.sched_setaffinity(self.proc_idx, self.affinity)

        # Load model
        from Dataset import Dataset

        self.data_obj = Dataset(self.dataset_param, self.mlperf_param)

        self.data_obj.load_dataset()
        log.info("Available samples: {} ".format(self.data_obj.count))

        if (self.warmup_count > 0):
            self.do_warmup()
        if self.num_workers > 1:
            start_core = self.start_core_idx
            cores_left = self.num_cores

            for i in range(self.num_workers):
                cores_left -= self.core_per_worker

                worker = mp.Process(target=self.handle_tasks, args=(i, self.task_queue, self.out_queue, self.pid))

                self.workers.append(worker)
                worker.start()
                start_core += self.core_per_worker

            for w in self.workers:
                w.join()
            log.info("{} : Exiting consumer process".format(os.getpid()))
        else:
            self.handle_tasks(0, self.task_queue, self.out_queue, self.pid)


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
        for id, out in zip(query_id_list, result):
            response_array = array.array(array_type_code, out)
            bi = response_array.buffer_info()
            responses = [lg.QuerySampleResponse(id, bi[0], bi[1] * response_array.itemsize)]
            lg.QuerySamplesComplete(responses)


def flush_queries():
    pass


def load_query_samples(query_samples):
    pass


def unload_query_samples(query_samples):
    pass


def main():
    args = get_args()
    with open(args.config, encoding='utf8') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    model_param = config["model_param"]
    dataset_param = config["dataset_param"]
    mlperf_param = config["mlperf_param"]
    system_param = config["system_param"]

    sys.path.insert(0, os.path.join(os.getcwd(), mlperf_param["workload"], system_param["backend"]))

    # Imports
    from InQueue import InQueue

    num_ins = system_param["num_instance"]
    num_cpus = system_param["core_per_instance"]

    # Establish communication queues
    lock = mp.Lock()
    init_counter = mp.Value("i", 0)

    settings = lg.TestSettings()
    settings.scenario = SCENARIO_MAP[mlperf_param["scenario"].lower()]
    settings.FromConfig(mlperf_param["mlperf_conf"], mlperf_param["workload"], mlperf_param["scenario"])
    settings.FromConfig(mlperf_param["user_conf"], mlperf_param["workload"], mlperf_param["scenario"])
    settings.mode = lg.TestMode.AccuracyOnly if mlperf_param["mode"].lower() == "accuracy" else lg.TestMode.PerformanceOnly

    print('setting', settings, type(settings))
    print(settings.offline_expected_qps)
    #return 

    consumers = []
    loadgen_cores = system_param["cores_offset"]

    out_queue = mp.Queue()

    input_queues = mp.JoinableQueue()

    i = 0
    while i < num_ins:
        start_core_idx = i * num_cpus + loadgen_cores
        consumer = Consumer(input_queues, out_queue, lock, init_counter, i, start_core_idx, model_param, system_param,
                            dataset_param, mlperf_param)
        consumers.append(consumer)
        i += 1

    # Update enqueue parameters and instantiate object        
    enqueueObj = InQueue(input_queues, dataset_param["batch_size"])

    for c in consumers:
        c.start()

    # Wait until all sub-processors are ready
    while init_counter.value < num_ins * system_param["num_worker_per_instance"]:
        time.sleep(2)

    # Start response thread
    resp_worker = threading.Thread(
        target=response_loadgen, args=(out_queue,))

    resp_worker.daemon = True
    resp_worker.start()

    def issue_queries(query_samples):
        enqueueObj.put(query_samples, receipt_time=time.time())

    sut = lg.ConstructSUT(
        issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        mlperf_param["total_sample_count"],
        min(mlperf_param["total_sample_count"], settings.performance_sample_count_override), load_query_samples,
        unload_query_samples)

    log_path = mlperf_param["output_logs"]
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    

    for _ in range(init_counter.value):
        input_queues.put(None)

    for c in consumers:
        c.join()

    out_queue.put(None)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


if __name__ == "__main__":
    start_time = time.time()
    main()
    log.info("Test completed")
    log.info("Test time {} s".format(time.time()-start_time))
    
