"""
implementation of criteo dataset
"""

# pylint: disable=unused-argument,missing-docstring

import os
import sys
import re
import random

import numpy as np
from torch_ipex import core
import inspect
# pytorch
import torch
from torch.utils.data import Dataset, RandomSampler
import os

# add dlrm code path
try:
    dlrm_dir_path = os.environ['DLRM_DIR']
    sys.path.append(dlrm_dir_path)
except KeyError:
    print("ERROR: Please set DLRM_DIR environment variable to the dlrm code location")
    sys.exit(0)
#import dataset
import dlrm_data_pytorch as dp
import data_loader_terabyte


class CriteoCalib(Dataset):

    def __init__(self,
                 data_path,
                 name,
                 test_num_workers=0,
                 max_ind_range=-1,
                 mlperf_bin_loader=False,
                 sub_sample_rate=0.0,
                 randomize="total",
                 memory_map=False):
        super().__init__()

        self.random_offsets = []
        self.use_fixed_size = True
        # fixed size queries
        self.samples_to_aggregate = 1

        if name == "kaggle":
            raw_data_file = data_path + "/train.txt"
            processed_data_file = data_path + "/kaggleAdDisplayChallenge_processed.npz"
        elif name == "terabyte":
            raw_data_file = data_path + "/day"
            processed_data_file = data_path + "/terabyte_processed.npz"
        else:
            raise ValueError("only kaggle|terabyte dataset options are supported")
        self.use_mlperf_bin_loader = mlperf_bin_loader and memory_map and name == "terabyte"

        if self.use_mlperf_bin_loader:
            cal_data_file = os.path.join(data_path, 'calibration.npz')
            if os.path.isfile(cal_data_file):
                print("Found calibration.npz !!")
                self.cal_loader = data_loader_terabyte.CalibDataLoader(
                    data_filename=cal_data_file,
                    batch_size=1,
                )
            else:
                counts_file = raw_data_file + '_fea_count.npz'
                validate_file = data_path + "/terabyte_processed_val.bin"
                if os.path.exists(validate_file):
                    print("Found terabyte_processed_val.bin !!")
                    self.val_data = data_loader_terabyte.CriteoBinDataset(
                        data_file=validate_file,
                        counts_file=counts_file,
                        batch_size=self.samples_to_aggregate,
                        max_ind_range=max_ind_range
                    )

                    self.val_loader = torch.utils.data.DataLoader(
                        self.val_data,
                        batch_size=None,
                        batch_sampler=None,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=None,
                        pin_memory=False,
                        drop_last=False,
                    )
                    self.cal_loader = self.val_loader
                else:
                    self.cal_loader = None
        else:
            self.val_data = dp.CriteoDataset(
                dataset=name,
                max_ind_range=max_ind_range,
                sub_sample_rate=sub_sample_rate,
                randomize=randomize,
                split="val",
                raw_path=raw_data_file,
                pro_data=processed_data_file,
                memory_map=memory_map
            )
            self.val_loader = torch.utils.data.DataLoader(
                self.val_data,
                batch_size=self.samples_to_aggregate,
                shuffle=False,
                num_workers=test_num_workers,
                collate_fn=dp.collate_wrapper_criteo,
                pin_memory=False,
                drop_last=False,
            )
            self.cal_loader = self.val_loader

    def get_calibration_data_loader(self):
        return self.cal_loader
