import array
import os
import sys
# import warnings
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
from squad_QSL import get_squad_QSL
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend")

from openvino.runtime import Core, get_version, AsyncInferQueue, PartialShape
from openvino.tools.benchmark.utils.constants import CPU_DEVICE_NAME

def get_version_info(self) -> str:
    log.info(f"OpenVINO:\n{'': <9}{'API version':.<24} {get_version()}")
    version_string = 'Device info\n'
    for device, version in self.core.get_versions(self.device).items():
        version_string += f"{'': <9}{device}\n"
        version_string += f"{'': <9}{version.description:.<24}{' version'} {version.major}.{version.minor}\n"
        version_string += f"{'': <9}{'Build':.<24} {version.build_number}\n"
    return version_string

class BERT_OpenVINO_SUT():
    def __init__(self, args):
        self.scenario = args.scenario
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.nireq = args.nireq
        self.inference_precision = args.inference_precision
        self.nstreams = args.nstreams
        self.nthreads = args.nthreads
        self.device = CPU_DEVICE_NAME
        if args.batch_size <= 0:
            raise ValueError("Batch size cannot be lower than or equal to 0")
        elif args.batch_size != 1 and args.scenario == 'Server':
            log.warn("Server scenario requires batch size of 1. Input batch size is ignored")
            self.batch_size = 1
        else:
            self.batch_size = args.batch_size

        self.core = Core()

        # Device Setup
        self.device_config = self._init_device_config()
        self._set_device_config(self.device_config)
        # self.print_device_property()

        # Model Setup
        # log.info(f"Loading OpenVINO model: {self.model_path}")
        self.compiled_model, self.input_port_names = self.load_model()

        # InferRequest Setup
        self.infer_queue = AsyncInferQueue(self.compiled_model, self.nireq)
        self.nireq = len(self.infer_queue)

        log.info("Core settings: {} infer requests, {} streams, {} threads".format(self.nireq,
            self.core.get_property(CPU_DEVICE_NAME, "NUM_STREAMS"),
            self.core.get_property(CPU_DEVICE_NAME, "INFERENCE_NUM_THREADS")))

        self.qsl = get_squad_QSL(args.max_examples)

        self.warmup_sut() # no advantage per experiments

        # log.info("Constructing SUT...")
        self.infer_queue.set_callback(self.callback)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        # log.info("Finished constructing SUT.")


    def load_model(self):
        loaded_model = self.core.read_model(self.model_path)
        seqlen=384
        new_shape_cfg = {}
        for iport in loaded_model.inputs:
            new_shape_cfg[iport.any_name] = PartialShape([self.batch_size, seqlen])
        loaded_model.reshape(new_shape_cfg)

        compiled_model = self.core.compile_model(loaded_model, self.device)
        input_port_names = [iport.any_name for iport in compiled_model.inputs]
        return compiled_model, input_port_names

    def _init_device_config(self):
        config = {
                CPU_DEVICE_NAME :
                    dict(
                            PERF_COUNT='NO',
                            PERFORMANCE_HINT='THROUGHPUT',
                            # NUM_STREAMS='-1',
                            # INFERENCE_PRECISION_HINT="bf16",
                        )
                }
        if self.inference_precision is not None:
            config[CPU_DEVICE_NAME]['INFERENCE_PRECISION_HINT'] = str(self.inference_precision)

        if self.nthreads is not None:
            config[CPU_DEVICE_NAME]['INFERENCE_NUM_THREADS'] = str(self.nthreads)

        if self.nstreams is not None:
            config[CPU_DEVICE_NAME]['NUM_STREAMS'] = str(self.nstreams)

        config[CPU_DEVICE_NAME]['PERFORMANCE_HINT_NUM_REQUESTS'] = str(self.nireq)
        return config

    def _set_device_config(self, config = {}):
        for device in config.keys():
            self.core.set_property(device, config[device])

    def print_device_property(self):
        keys = self.core.get_property(CPU_DEVICE_NAME, 'SUPPORTED_PROPERTIES')
        log.info(f'DEVICE: {CPU_DEVICE_NAME}')
        for k in keys:
            if k not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
                try:
                    log.info(f' {k}  , {self.core.get_property(CPU_DEVICE_NAME, k)}')
                except:
                    pass

    def warmup_sut(self):
        WARMUP_CYCLES = 100
        for raw_input in self.qsl.eval_features[::WARMUP_CYCLES]:
            self.infer_queue.start_async(inputs=[raw_input.input_ids, raw_input.input_mask, raw_input.segment_ids])
        self.infer_queue.wait_all()

    def issue_queries(self, query_samples):
        for query_sample in query_samples:
            raw_input = self.qsl.get_features(query_sample.index)
            self.infer_queue.start_async(inputs=[raw_input.input_ids, raw_input.input_mask, raw_input.segment_ids], userdata=query_sample.id)

    def flush_queries(self):
        self.infer_queue.wait_all()

    def callback(self, infer_request, userdata):
        scores = list(infer_request.results.values())
        output = np.stack(scores, axis=-1)

        response_array = array.array("B", output.tobytes())
        bi = response_array.buffer_info()
        response = lg.QuerySampleResponse(userdata, bi[0], bi[1])
        lg.QuerySamplesComplete([response])

    def __del__(self):
        log.info("Finished destroying SUT.")

def get_openvino_sut(args):
    return BERT_OpenVINO_SUT(args)
