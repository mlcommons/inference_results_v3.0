import sys
import os
import logging
import time
import array
import json

import mlperf_loadgen as lg
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from kits_QSL import get_kits_QSL

from InputData import InputData
from OutputItem import OutputItem

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("3DUnet-Dataset")


class Dataset(object):
    # count
    # load_query_samples
    # unload_query_samples
    def __init__(self, data_path=None, performance_count=16, verbose=True, **kwargs):
        super().__init__()
        self.qsl = None
        self.count = 0
        self.data_path = data_path
        self.performance_count = performance_count
        self.verbose = verbose
        self.load_dataset()

        self.sample_id_subcnt = {
            0:50, 1:50, 2:108, 3:64, 4:48, 5:27, 6:100, 7:16, 8:100, 9:50, 10:50, 
            11:48, 12:36, 13:32, 14:100, 15:32, 16:32, 17:50, 18:48, 19:144, 20:64, 
            21:16, 22:64, 23:64, 24:16, 25:144, 26:100, 27:16, 28:50, 29:8, 30:50, 
            31:36, 32:64, 33:80, 34:144, 35:125, 36:27, 37:96, 38:96, 39:45, 40:50, 41:96, 42:125
        }


    def load_dataset_into_memory(self):
        pass

    def load_dataset(self):
        if self.qsl == None:
            self.qsl = get_kits_QSL(self.data_path, self.performance_count)
            self.count = self.qsl.count

    def get_qsl(self):
        self.load_dataset()
        return self.qsl


    def load_query_samples(self, sample_list):
        self.qsl.load_query_samples(sample_list)

    def unload_query_samples(self, sample_list):
        self.qsl.unload_query_samples(sample_list)

    def get_warmup_samples(self):
        import random
        samples = []
        num_samples = 1
        for _ in range(num_samples):
            samples.append(self.get_qsl().get_features(random.randint(0,self.count-1)))
        return InputData(data=samples)

    def get_samples(self, sample_index_list):
        batch_size = len(sample_index_list)
        data = []
        for i in range(batch_size):
            data.append(self.get_qsl().get_features(sample_index_list[i]))
            # if self.verbose:
            #     print("Processing sample id {:d} with shape = {:}, sub_cnt = {:d}".format(sample_index_list[i], list(data[i].shape), self.sample_id_subcnt[sample_index_list[i]]))
        return InputData(data=data)


    def post_process(self, query_ids, sample_index_list, results):
        return OutputItem(query_ids, results, array_type_code='B')