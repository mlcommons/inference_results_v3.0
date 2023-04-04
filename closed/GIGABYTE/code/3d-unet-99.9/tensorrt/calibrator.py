# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from pathlib import Path

from code.common import logging
from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import tensorrt as trt
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoprimaryctx


class UNet3DKiTS19MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, data_dir, cache_file, batch_size, max_batches,
                 force_calibration, calib_data_map):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.cache_file = cache_file
        self.batch_size = batch_size
        self.max_batches = max_batches

        vol_list = list()
        vol_list = []
        with open(calib_data_map) as f:
            for line in f:
                vol_list.append(line.strip())
        vol_list = sorted(vol_list)

        self.kits_id = 0
        self.batch_id = 0

        self.force_calibration = force_calibration

        self.device_input = None

        def load_batches_full_image():
            """
            Create a generator that will give us batches. We can use next() to iterate over the result.
            Only supports batch_size == 1 for the KiTS19 dataset, since KiTS19 dataset has non-uniform input dimensions.
            """

            # enforce 1 batch
            if self.batch_size != 1:
                print("WARNING: Calibrating with KiTS19 full-image limits batchsize of 1 due to nonuniform input shape. "
                      "Overriding batchsize to 1 ")
                self.batch_size = 1

            while self.kits_id < len(vol_list) and self.kits_id < self.max_batches:
                image = np.load(Path(data_dir, vol_list[self.kits_id] + ".npy"))
                image = np.ascontiguousarray(image[np.newaxis, ...])

                print("Calibrating with image_ID: {} of name: {}".format(
                    self.kits_id, vol_list[self.kits_id]))

                self.kits_id += 1

                yield image

        self.batches = load_batches_full_image()

        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None
        if not self.force_calibration and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.cache = f.read()
        else:
            self.cache = None

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """
        Acquire a single batch 

        Arguments:
        names (string): names of the engine bindings from TensorRT. Useful to understand the order of inputs.
        """
        try:
            # get next batch -- contiguous numpy array of float32
            data = next(self.batches)
            # Copy to device
            self.device_input = cuda.to_device(data)
            cuda.memcpy_htod(self.device_input, data.tobytes())
            # return pointer to, i.e. address of, this device buffer holding calibration batch
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, return [].
            # This signals to TensorRT that there is no calibration data remaining
            return []

    def read_calibration_cache(self):
        return self.cache

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def clear_cache(self):
        self.cache = None
