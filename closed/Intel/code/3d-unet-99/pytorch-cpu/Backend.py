import sys
import os
import logging
import time
import array
import json

from pathlib import Path
# import mlperf_loadgen as lg
import inference_utils as infu
from global_vars import *

import numpy as np
import torch
import torch.nn.functional as F
# import torch.autograd.profiler as profiler

import intel_pytorch_extension as ipex
from baseBackend import baseBackend
from unet3d_kits_model import Unet3D

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("3DUnet-Backend")

import torch.autograd.profiler as profiler

class Backend(baseBackend):
    def __init__(self, model_path=None, folds=1, checkpoint_name="model_final_checkpoint", verbose=True, calibrate=False, calibration_file="calibration_result.json", *kwargs):
        self.model_path = model_path
        assert Path(model_path).is_file(), "Cannot find the model file {:}!".format(model_path)

        self.verbose = verbose
        self.total_slice_image = 0
        self.profiler = profiler.profile(record_shapes=True)

        self.int8_conf = None
        if not calibrate:
            if not os.path.isfile(calibration_file):
                log.error("Cannot find int8 calibration file {}".format(calibration_file))
                sys.exit(1)
            self.int8_conf = ipex.AmpConf(torch.int8, calibration_file)


    def load_model(self):
        if self.verbose:
            print("Loading PyTorch model...")
        self.model = torch.jit.load(self.model_path)
        if self.verbose:
            print("Loading PyTorch model...")   
   

    def predict(self, input_data):
        outputs = self.fw_inference(input_data)

        responses = []
        for i in range(len(input_data)):
            responses.append(array.array("B", outputs[i].tobytes()))
        return responses

    def fw_inference(self, data ,calibrate_flag=False, conf=None):
        output = []
        for query in data:
            output.append(self.infer_single_query_preconditioned(query, calibrate_flag, conf))
        return output

    def infer_single_query(self, query, calibrate_flag, conf):
        """
        Performs inference upon data and summarize work with mystr
        Naive implementation of sliding window inference on sub-volume for predetermined
        ROI (Region of Interest) shape is handled here
        Parameters
        ----------
            query: object
                Query sent by LoadGen
        """
        # prepare arrays
        image = query
        result, norm_map, norm_patch = infu.prepare_arrays(image, ROI_SHAPE)
        t_norm_patch = self.to_tensor(norm_patch)

        # sliding window inference
        subvol_cnt = 0
        for i, j, k, _ in infu.get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
            subvol_cnt += 1
            result_slice = result[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            input_slice = image[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            norm_map_slice = norm_map[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            p_num = -1
            if subvol_cnt == p_num:
                self.profiler.__enter__()
                print("--------begin profiler---------")
            
            # t1 = time.time()
            result_slice += self.do_infer(input_slice, calibrate_flag, conf) * t_norm_patch
            # print("time is ", time.time() - t1, "sub_cnt", subvol_cnt)

            if subvol_cnt == p_num:
                self.profiler.__exit__(None, None, None)
                self.profile_print(self.profiler, "int8_inference.json")
                # train_profiler.export_chrome_trace('p_fp32_chrome.json')
                print("--------profiler result has been saved---------")

            norm_map_slice += t_norm_patch

        self.total_slice_image += subvol_cnt
        # print("total slice image ", self.total_slice_image)
        result, norm_map = self.from_tensor(
            result), self.from_tensor(norm_map)

        final_result = infu.finalize(result, norm_map)
        return final_result

    def do_infer(self, input_tensor, calibrate_flag, conf):
        """
        Perform inference upon input_tensor with PyTorch/TorchScript
        """         
        if calibrate_flag:
            with torch.no_grad():
                with ipex.AutoMixPrecision(conf, running_mode='calibration'):
                    return self.model(input_tensor.to(ipex.DEVICE))
        with torch.no_grad():
            with ipex.AutoMixPrecision(self.int8_conf, running_mode='inference'):
                return self.model(input_tensor.to(ipex.DEVICE))

    def to_tensor(self, my_array):
        """
        Transform numpy array into Torch tensor
        """
        return torch.from_numpy(my_array).float()

    def from_tensor(self, my_tensor):
        """
        Transform Torch tensor into numpy array
        """
        return my_tensor.cpu().numpy().astype(np.float)


    def calibrate(self, samples, qsl, conf_json_file):
        conf = ipex.AmpConf(torch.int8)

        batch_size = len(samples)
        for i in range(batch_size):
            data = []
            data.append(qsl.get_features(samples[i]))
            if self.verbose:
                print("[{:}/{:}] Calibrating sample id {:d} with shape = {:}".format(i, batch_size, samples[i], data[0].shape))
            output = self.fw_inference(data, calibrate_flag=True, conf=conf)
        conf.save(conf_json_file)
        print('calibration_result saved')
        if self.verbose:
            print("Calibration configuration saved to {}".format(conf_json_file))
        self.int8_conf = conf


    def profile_print(self, profiler, file=None):
        if file == None:
            print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
            print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            print(profiler.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=40))
            print(profiler.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=40))
        else:
            with open(file, 'w', encoding = 'utf-8') as f:
                f.write(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
                f.write(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
                f.write(profiler.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=40))
                f.write(profiler.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=40))


    def infer_single_query_preconditioned(self, query, calibrate_flag, conf):
        """
        Performs inference upon data and summarize work with mystr
        Naive implementation of sliding window inference on sub-volume for predetermined
        ROI (Region of Interest) shape is handled here
        Parameters
        ----------
            query: object
                Query sent by LoadGen
        """
        # prepare arrays
        image = query
        result, norm_patches = infu.prepare_arrays_preconditioned(image, ROI_SHAPE)

        # sliding window inference
        for i, j, k, patch_key, _ in infu.get_slice_for_sliding_window_preconditioned(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
            t_norm_patch = norm_patches[patch_key]

            result_slice = result[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            input_slice = image[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            result_slice += self.do_infer(input_slice, calibrate_flag, conf) * t_norm_patch

        result= self.from_tensor(result)
        final_result = infu.finalize(result, None)
        return final_result


