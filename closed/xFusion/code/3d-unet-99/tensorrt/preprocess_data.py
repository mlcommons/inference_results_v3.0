#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__doc__ = """
Preprocess data for 3D-UNet KiTS19 benchmark.
Numpy files, in formats compatible for TensorRT, are created as end results

Example command:
    python code/3d-unet/tensorrt/preprocess_data.py --data_dir build/data --preprocessed_data_dir build/preprocessed_data

"""

import argparse
import os
import json
import pickle
from collections import OrderedDict
from multiprocessing import Process, Pool
from pathlib import Path
from itertools import repeat
from typing import Dict, List, Tuple, Union, Generator

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np
    import nibabel
    from scipy.ndimage.interpolation import zoom
    from scipy import signal

from code.common.utils import get_dyn_ranges


class KITS19Tool:
    """
    A class storing many constants and pointers for (pre-)processing KiTS19 dataset

    Attributes
    ----------
    PREPROCESSED_DATA_DIR, PREPROCESSED_REF_DIR, PREPROCESSED_INFER_DIR, PREPROCESSED_CALIB_DIR: str
        directories preprocessed data, for reference pickle, inference and calibration numpy files are stored into
    KITS_DATA_DIR, KITS_RAW_DIR: str
        directories containing kits19 dataset repository and the downloaded KiTS19 RAW data
    INFERENCE_CASE_FILE, CALIBRATION_CASE_FILE: str
        files containing what cases are used for inference and calibration purposes
    INFER_CASES, CALIB_CASES: [str, str, ...]
        lists containing case numbers in string that are used for inference and calibration purposes
    MEAN_VAL, STDDEV_VAL, MIN_CLIP_VAL, MAX_CLIP_VAL: float
        used for normalizing intensity of the input image
    PADDING_VAL: float
        padding with this value
    TARGET_SPACING: [z, y, x]
        common voxel spacing that all the CT images reshaped for
    ROI_SHAPE: [d, h, w]
        ROI (Region of Interest) shape; training done on the sub-volume of this shape
    SLIDE_OVERLAP_FACTOR: float
        sliding window inference will follow this overlapping factor
    MULTI_POOL: obj
        pool object that are used for multiprocessing

    Methods
    -------
    __init__():
        initiates all the attributes
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initiates all the attributes

        Attributes
        ----------
        PREPROCESSED_DATA_DIR, PREPROCESSED_REF_DIR, PREPROCESSED_INFER_DIR, PREPROCESSED_CALIB_DIR: str
            directories preprocessed data, for reference pickle, inference and calibration numpy files are stored into
        KITS_DATA_DIR, KITS_RAW_DIR: str
            directories containing kits19 dataset repository and the downloaded KiTS19 RAW data
        INFERENCE_CASE_FILE, CALIBRATION_CASE_FILE: str
            files containing what cases are used for inference and calibration purposes
        INFER_CASES, CALIB_CASES: [str, str, ...]
            lists containing case numbers in string that are used for inference and calibration purposes
        MEAN_VAL, STDDEV_VAL, MIN_CLIP_VAL, MAX_CLIP_VAL: float
            used for normalizing intensity of the input image
        PADDING_VAL: float
            padding with this value
        TARGET_SPACING: [z, y, x]
            common voxel spacing that all the CT images reshaped for
        ROI_SHAPE: [d, h, w]
            ROI (Region of Interest) shape; training done on the sub-volume of this shape
        SLIDE_OVERLAP_FACTOR: float
            sliding window inference will follow this overlapping factor
        MULTI_POOL: obj
            pool object that are used for multiprocessing
        """
        # file pointers and sanity checks
        self.KITS_DATA_DIR = Path(args.data_dir, 'KiTS19').absolute()
        self.KITS_RAW_DIR = Path(self.KITS_DATA_DIR, 'kits19', 'data').absolute()
        self.PREPROCESSED_DATA_DIR = Path(args.preprocessed_data_dir, 'KiTS19').absolute()
        self.PREPROCESSED_REF_DIR = Path(self.PREPROCESSED_DATA_DIR, 'reference').absolute()
        self.PREPROCESSED_INFER_DIR = Path(self.PREPROCESSED_DATA_DIR, 'inference').absolute()
        self.PREPROCESSED_CALIB_DIR = Path(self.PREPROCESSED_DATA_DIR, 'calibration').absolute()
        self.PREPROCESSED_ETC_DIR = Path(self.PREPROCESSED_DATA_DIR, 'etc').absolute()
        self.INFERENCE_CASE_FILE = Path(self.KITS_DATA_DIR, 'inference_cases.json').absolute()
        self.CALIBRATION_CASE_FILE = Path(self.KITS_DATA_DIR, 'calibration_cases.json').absolute()
        assert self.INFERENCE_CASE_FILE.is_file(), 'inference_cases.json is not found'
        assert self.CALIBRATION_CASE_FILE.is_file(), 'calibration_cases.json is not found'

        # cases used for inference and calibration
        self.INFER_CASES = json.load(open(self.INFERENCE_CASE_FILE))
        self.CALIB_CASES = json.load(open(self.CALIBRATION_CASE_FILE))

        # constants used preprocessing images as well as sliding window inference
        self.MEAN_VAL = 101.0
        self.STDDEV_VAL = 76.9
        self.MIN_CLIP_VAL = -79.0
        self.MAX_CLIP_VAL = 304.0
        self.PADDING_VAL = -2.2
        self.TARGET_SPACING = [1.6, 1.2, 1.2]
        self.ROI_SHAPE = [128, 128, 128]
        self.SLIDE_OVERLAP_FACTOR = 0.5
        assert isinstance(self.TARGET_SPACING, list) and \
            len(self.TARGET_SPACING) == 3 and any(self.TARGET_SPACING), \
            "Need proper target spacing: {}".format(self.TARGET_SPACING)
        assert isinstance(self.ROI_SHAPE, list) and len(self.ROI_SHAPE) == 3 and \
            any(self.ROI_SHAPE), \
            "Need proper ROI shape: {}".format(self.ROI_SHAPE)
        assert isinstance(self.SLIDE_OVERLAP_FACTOR, float) and \
            self.SLIDE_OVERLAP_FACTOR > 0 and self.SLIDE_OVERLAP_FACTOR < 1, \
            "Need sliding window overlap factor in (0,1): {}".format(self.SLIDE_OVERLAP_FACTOR)

        # for multiprocessing
        self.MULTI_POOL = Pool(args.num_proc)


class Preprocessor:
    """
    A class processing images in KiTS19 dataset
    Pre-processing includes below steps (128x128x128 window with 50% overlap as an example)
        1. Get a pair of CT-imaging/segmentation data
        2. Resample to the same, predetermined common voxel spacing (1.6, 1.2, 1.2)[mm]
        3. Pad every volume so it is equal or larger than 128
        4. Pad/crop volumes so they are divisible by 64
    Preprocessed data are saved as pickle format for easy consumption
    Reshaped imaging/segmentation will be saved as NIFTI as well for easy comparison with prediction

    Attributes
    ----------
    results_dir: str
        directory preprocessed data will be stored into
    data_dir: str
        directory containing KiTS19 RAW data
    calibration: bool
        flag for processing calibration set, if true, instead of inference set
    mean, std, min_val, max_val: float
        used for normalizing intensity of the input image
    padding_val: float
        padding with this value
    target_spacing: [z, y, x]
        common voxel spacing that all the CT images reshaped for
    target_shape: [d, h, w]
        ROI (Region of Interest) shape; training done on the sub-volume of this shape
    slide_overlap_factor: float
        sliding window inference will follow this overlapping factor

    Methods
    -------
    __init__():
        initiates all the attributes
    collect_cases():
        populates cases to preprocess from attribute target_cases
    preprocess_dataset():
        performs preprocess of all the cases collected
    preprocess_case(case):
        picks up the case from KiTS19 RAW data and perform preprocessing:
        1. Get a pair of CT-imaging/segmentation data for the case
        2. Resample to the same, predetermined common voxel spacing
        3. Pad every volume so it is equal or larger than ROI shape
        4. Pad/Crop volumes so they are friendly to sliding window inference
        then save the preprocessed data
    pad_to_min_shape(image, label):
        pads image/label so that the shape is equal or larger than ROI shape
    load_and_resample(case):
        gets a pair of CT-imaging/segmentation data for the case, then, 
        resample to the same, predetermined common voxel spacing
    normalize_intensity(image):
        normalize intensity for a given target stats
    adjust_shape_for_sliding_window(image, label):
        pads/crops image/label volumes so that sliding window inference can easily be done
    constant_pad_volume(volume, roi_shape, strides, padding_val, dim):
        helper padding volume symmetrically with value of padding_val
        padded volume becomes ROI shape friendly
    save(image, label, aux):
        Save preprocessed imaging/segmentation data in pickle format for easy consumption
        auxiliary information also saved together that holds:
    """

    def __init__(self, kits19tool: KITS19Tool) -> None:
        """
        Initiates all the attributes

        Attributes
        ----------
        results_dir: str
            directory preprocessed data will be stored into
        data_dir: str
            directory containing KiTS19 RAW data
        infer_cases, calib_cases, target_cases: [str, str, ...]
            list containing strings pointing KiTS19 cases, for inference, calibration and both
        mean, std, min_val, max_val: float
            used for normalizing intensity of the input image
        padding_val: float
            padding with this value
        target_spacing: [z, y, x]
            One common voxel spacing that all the CT images reshaped for
        target_shape: [d, h, w]
            ROI (Region of Interest) shape; training done on the sub-volume of this shape
        slide_overlap_factor: float
            sliding window inference will follow this overlapping factor
        """
        self.results_dir = str(kits19tool.PREPROCESSED_REF_DIR)
        self.data_dir = str(kits19tool.KITS_RAW_DIR)
        self.infer_cases = kits19tool.INFER_CASES
        self.calib_cases = kits19tool.CALIB_CASES
        self.target_cases = sorted(self.infer_cases + self.calib_cases)
        self.mean = kits19tool.MEAN_VAL
        self.std = kits19tool.STDDEV_VAL
        self.min_val = kits19tool.MIN_CLIP_VAL
        self.max_val = kits19tool.MAX_CLIP_VAL
        self.padding_val = kits19tool.PADDING_VAL
        self.target_spacing = kits19tool.TARGET_SPACING
        self.target_shape = kits19tool.ROI_SHAPE
        self.slide_overlap_factor = kits19tool.SLIDE_OVERLAP_FACTOR
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

    def collect_cases(self) -> List:
        """
        Populates cases to preprocess from attribute target_cases
        """
        print(f"Preprocessing {self.data_dir}...")
        all_set = set([f for f in os.listdir(self.data_dir) if "case" in f])
        target_set = set(self.target_cases)
        collected_set = all_set & target_set
        assert collected_set == target_set,\
            "Some of the target inference cases were NOT found: {}".format(
                target_set - collected_set)
        return sorted(list(collected_set))

    def preprocess_dataset(self) -> None:
        """
        Performs preprocess of all the cases collected
        """
        for case in self.collect_cases():
            self.preprocess_case(case)

    def preprocess_case(self, case: str) -> Dict:
        """
        Picks up the case from KiTS19 RAW data and perform preprocessing:
            1. Get a pair of CT-imaging/segmentation data for the case
            2. Resample to the same, predetermined common voxel spacing (1.6, 1.2, 1.2)[mm]
            3. Pad every volume so it is equal or larger than 128
            4. Pad/Crop volumes so they are divisible by 64
        Then save the preprocessed data in pickle format for easy consumption
        Reshaped imaging/segmentation will be saved as NIFTI as well for easy comparison with prediction
        """
        image, label, aux = self.load_and_resample(case)
        image = self.normalize_intensity(image.copy())
        image, label = self.pad_to_min_shape(image, label, self.target_shape)
        image, label = self.adjust_shape_for_sliding_window(image, label)
        self.save(image, label, aux)
        aux['image_shape'] = image.shape
        return aux

    @staticmethod
    def pad_to_min_shape(image: np.ndarray,
                         label: np.ndarray,
                         roi_shape: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pads every volume so it is equal or larger than ROI shape
        """
        current_shape = image.shape[1:]
        bounds = [max(0, roi_shape[i] - current_shape[i]) for i in range(3)]
        paddings = [(0, 0)]
        paddings.extend([(bounds[i] // 2, bounds[i] - bounds[i] // 2)
                         for i in range(3)])

        image = np.pad(image, paddings, mode="edge")
        label = np.pad(label, paddings, mode="edge")

        return image, label

    def load_and_resample(self, case: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Gets a pair of CT-imaging/segmentation data for the case
        Then, resample to the same, predetermined common voxel spacing (1.6, 1.2, 1.2)[mm]
        Also store auxiliary info for future use
        """
        aux = dict()

        image = nibabel.load(
            Path(self.data_dir, case, "imaging.nii.gz").absolute())
        label = nibabel.load(
            Path(self.data_dir, case, "segmentation.nii.gz").absolute())

        image_spacings = image.header["pixdim"][1:4].tolist()
        original_affine = image.affine

        image = image.get_fdata().astype(np.float32)
        label = label.get_fdata().astype(np.uint8)

        spc_arr = np.array(image_spacings)
        targ_arr = np.array(self.target_spacing)
        zoom_factor = spc_arr / targ_arr

        # build reshaped affine
        reshaped_affine = original_affine.copy()
        for i in range(3):
            idx = np.where(original_affine[i][:-1] != 0)
            sign = -1 if original_affine[i][idx] < 0 else 1
            reshaped_affine[i][idx] = targ_arr[idx] * sign

        if image_spacings != self.target_spacing:
            image = zoom(image, zoom_factor, order=1,
                         mode='constant', cval=image.min(), grid_mode=False)
            label = zoom(label, zoom_factor, order=0,
                         mode='constant', cval=label.min(), grid_mode=False)

        aux['original_affine'] = original_affine
        aux['reshaped_affine'] = reshaped_affine
        aux['zoom_factor'] = zoom_factor
        aux['case'] = case

        image = np.expand_dims(image, 0)
        label = np.expand_dims(label, 0)

        return image, label, aux

    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizes intensity for a given target stats
        """
        image = np.clip(image, self.min_val, self.max_val)
        image = (image - self.mean) / self.std
        return image

    def adjust_shape_for_sliding_window(self,
                                        image: np.ndarray,
                                        label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pads/crops volumes so that sliding window inference can easily be done

        Sliding window of 128x128x128 to move smoothly, with overlap factor of 0.5
        then pads/crops volumes so that they are divisible by 64
        This padding or cropping is done as below:
            - if a given edge length modulo 64 is larger than 32 it is constant padded
            - if a given edge length modulo 64 is less than 32 it will be cropped
        """
        image_shape = list(image.shape[1:])
        dim = len(image_shape)
        roi_shape = self.target_shape
        overlap = self.slide_overlap_factor
        strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

        bounds = [image_shape[i] % strides[i] for i in range(dim)]
        bounds = [bounds[i] if bounds[i] <
                  strides[i] // 2 else 0 for i in range(dim)]
        image = image[...,
                      bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
                      bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
                      bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
        label = label[...,
                      bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
                      bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
                      bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
        image, paddings = self.constant_pad_volume(
            image, roi_shape, strides, self.padding_val)
        label, paddings = self.constant_pad_volume(
            label, roi_shape, strides, 0)

        return image, label

    def constant_pad_volume(self,
                            volume: np.ndarray,
                            roi_shape: List,
                            strides: List,
                            padding_val: List,
                            dim: int = 3) -> Tuple[np.ndarray, List]:
        """
        Helper padding volume symmetrically with value of padding_val
        Padded volume becomes ROI shape friendly
        """
        bounds = [(strides[i] - volume.shape[1:][i] % strides[i]) %
                  strides[i] for i in range(dim)]
        bounds = [bounds[i] if (volume.shape[1:][i] + bounds[i]) >= roi_shape[i] else
                  bounds[i] + strides[i]
                  for i in range(dim)]
        paddings = [(0, 0),
                    (bounds[0] // 2, bounds[0] - bounds[0] // 2),
                    (bounds[1] // 2, bounds[1] - bounds[1] // 2),
                    (bounds[2] // 2, bounds[2] - bounds[2] // 2)]

        padded_volume = np.pad(
            volume, paddings, mode='constant', constant_values=[padding_val])
        return padded_volume, paddings

    def save(self, image: np.ndarray, label: np.ndarray, aux: Dict) -> None:
        """
        Saves preprocessed imaging/segmentation data in pickle format for easy consumption
        Auxiliary information also saved together that holds:
            - preprocessed image/segmentation shape
            - original affine matrix
            - affine matrix for reshaped imaging/segmentation upon common voxel spacing
            - zoom factor used in transform from original voxel spacing to common voxel spacing
            - case name
        Preprocessed imaging/segmentation data saved as NIFTI
        """
        case = aux['case']
        reshaped_affine = aux['reshaped_affine']
        image = image.astype(np.float32)
        label = label.astype(np.uint8)
        mean, std = np.round(np.mean(image, (1, 2, 3)), 2), np.round(
            np.std(image, (1, 2, 3)), 2)
        pickle_file_path = Path(self.results_dir, f"{case}.pkl").absolute()
        with open(pickle_file_path, "wb") as f:
            pickle.dump([image, label], f)
        f.close()
        print(
            f"Saved {str(pickle_file_path)} -- shape {image.shape} mean {mean} std {std}")
        path_to_nifti_dir = Path(
            self.results_dir, "nifti", case).absolute()
        path_to_nifti_dir.mkdir(parents=True, exist_ok=True)
        nifti_image = nibabel.Nifti1Image(
            np.squeeze(image, 0), affine=reshaped_affine)
        nifti_label = nibabel.Nifti1Image(
            np.squeeze(label, 0), affine=reshaped_affine)
        nibabel.save(nifti_image, Path(
            path_to_nifti_dir / "imaging.nii.gz"))
        nibabel.save(nifti_label, Path(
            path_to_nifti_dir / "segmentation.nii.gz"))
        assert nifti_image.shape == nifti_label.shape, \
            "While saving NIfTI files to {}, image: {} and label: {} have different shape".format(
                path_to_nifti_dir, nifti_image.shape, nifti_label.shape)


def preprocess_ref_multiproc_helper(preproc: Preprocessor, case: str) -> Dict:
    """
    Helps preprocessing reference dataset with multi-processes
    """
    aux = preproc.preprocess_case(case)
    return aux


def save_preprocessed_ref_info(preproc_dir: str, aux: Dict, targets: List) -> None:
    """
    Saves list of preprocessed reference dataset files and the associated aux info into preprocessed_files.pkl
    """
    assert len(targets) == len(aux['cases']),\
        "Error in number of preprocessed files:\nExpected:{}\nProcessed:{}".format(
            targets, list(aux['cases'].keys()))
    with open(Path(preproc_dir, 'preprocessed_files.pkl'), 'wb') as f:
        pickle.dump(aux, f)
    f.close()


def preprocess_ref_with_multiproc(kits19tool: KITS19Tool) -> None:
    """
    Performs preprocess on KiTS19 imaging/segmentation data using multiprocesses
    """
    preproc = Preprocessor(kits19tool)
    cases = preproc.collect_cases()
    aux = {
        'eval_list': preproc.infer_cases,
        'file_list': preproc.target_cases,
        'cases': dict()
    }
    pool_out = kits19tool.MULTI_POOL.starmap(preprocess_ref_multiproc_helper,
                                             zip([preproc] * len(cases), cases))

    for _d in pool_out:
        aux['cases'][_d['case']] = _d
    save_preprocessed_ref_info(preproc.results_dir, aux, preproc.target_cases)


def preprocess_kits19_raw_data(kits19tool: KITS19Tool) -> None:
    """
    Preprocesses KiTS19 RAW data, for both image and segmentation
        - Resample to a common voxel spacing
        - Pad every volume so it is equal or larger than 128
        - Crop/Pad volumes so they are divisible by 64
    """
    preprocess_ref_with_multiproc(kits19tool)


def get_dynamic_range_from_cache(cachefile: Union[str, os.PathLike], tensor_name: str) -> float:
    """
    Read calibrator cache file and return the dynamic range of tensor identified by its name
    If tensor name is not found, return 0
    """
    dynamic_range_dict = get_dyn_ranges(cachefile)
    dr = dynamic_range_dict.get(tensor_name, 0)
    return dr


def clamp_to_int8(f):
    # Round half to even first
    rounded_result = 0.0
    r = round(f)
    d = r - f
    if d != 0.5 and d != -0.5:
        rounded_result = r
    elif r % 2.0 == 0.0:
        rounded_result = r
    else:
        rounded_result = f - d
    return max(-128.0, min(127.0, rounded_result))


def get_3dunet_int8_linear(fp32_input: np.ndarray) -> np.ndarray:
    """
    Helper function to convert input data into int8 Linear format and return it
    """
    base_dir = Path(__file__).absolute().parent
    cache_file = Path(base_dir, 'calibrator.cache')
    if not cache_file.is_file():
        print("calibration.cache file does not exist - please retry after calibration.")
        # scale cannot be determined so give up and return 0 size array
        return np.array([])

    input_dr = get_dynamic_range_from_cache(cache_file, 'input')
    int8_input = np.round(fp32_input / input_dr * np.iinfo(np.int8).max).astype(np.int8)

    return int8_input


def get_3dunet_int8_cdhw32(fp32_input: np.ndarray) -> np.ndarray:
    """
    Helper function to convert input data into int8 C/32DHW32 format, returning 32ch zero-padded tensor
    """
    base_dir = Path(__file__).absolute().parent
    cache_file = Path(base_dir, 'calibrator.cache')
    if not cache_file.is_file():
        print("calibration.cache file does not exist - please retry after calibration.")
        # scale cannot be determined so give up and return 0 size array
        return np.array([])

    input_dr = get_dynamic_range_from_cache(cache_file, 'input')
    int8_input = np.round(fp32_input / input_dr * np.iinfo(np.int8).max).astype(np.int8)

    return np.pad(int8_input, ((0, 31), (0, 0), (0, 0), (0, 0)), mode="constant").transpose(1, 2, 3, 0)


def get_3dunet_fp16_linear(fp16_input: np.ndarray) -> np.ndarray:
    """
    Return fp16 linear tensor
    """
    return fp16_input


def get_3dunet_fp16_dhwc8(fp16_input: np.ndarray) -> np.ndarray:
    """
    Return 8ch zero-padded fp16 DHWC8 tensor
    """
    return np.pad(fp16_input,
                  ((0, 7), (0, 0), (0, 0), (0, 0)),
                  mode="constant").transpose(1, 2, 3, 0)


def preprocess_numpy_multiproc_helper(tgt_case: str,
                                      tgt_str: str,
                                      ref_dir: Union[str, os.PathLike],
                                      fp32_dir: Union[str, os.PathLike],

                                      fp16_dhwc8_dir: Union[str, os.PathLike],
                                      int8_cdhw32_dir: Union[str, os.PathLike],

                                      fp16_linear_dir: Union[str, os.PathLike],
                                      int8_linear_dir: Union[str, os.PathLike]):
    """
    Helps generating Numpy files from preprocessed ref dataset with multi-processes
    """

    # read preprocessed ref file
    with open(Path(ref_dir, f"{tgt_case}.pkl"), "rb") as f:
        # d(ata) and s(egmentation)
        d, s = pickle.load(f)

    # simple sanity check for input tensor
    assert d.shape == s.shape, "{}: preprocessed ref data seems not right".format(tgt_case)
    assert [i % 64 for i in d.shape] == [1, 0, 0, 0], "{}: Unexpected tensor shape: {}".format(tgt_case, d.shape)

    # save after conversion
    np.save(Path(fp32_dir, f"{tgt_case}.npy"), d.astype(np.float32))
    if tgt_str != 'calib':
        fp16_linear = get_3dunet_fp16_linear(d.astype(np.float16))
        np.save(Path(fp16_linear_dir, f"{tgt_case}.npy"), fp16_linear)

        fp16_dhwc8 = get_3dunet_fp16_dhwc8(d.astype(np.float16))
        np.save(Path(fp16_dhwc8_dir, f"{tgt_case}.npy"), fp16_dhwc8)

        int8_linear = get_3dunet_int8_linear(d.astype(np.float32))
        if int8_linear.size > 0:
            np.save(Path(int8_linear_dir, f"{tgt_case}.npy"), int8_linear)

        int8_cdhw32 = get_3dunet_int8_cdhw32(d.astype(np.float32))
        if int8_cdhw32.size > 0:
            np.save(Path(int8_cdhw32_dir, f"{tgt_case}.npy"), int8_cdhw32)

    print(f"{tgt_case} converted.")


def preprocess_numpy_with_multiproc(kits19tool: KITS19Tool, work_package: Dict) -> None:
    """
    Generates Numpy files from preprocessed ref dataset with multi-processes
    """
    ref_dir = work_package['ref_dir']
    fp32_dir = work_package['fp32_dir']
    fp16_linear_dir = work_package['fp16_linear_dir']
    int8_linear_dir = work_package['int8_linear_dir']

    fp16_dhwc8_dir = work_package['fp16_dhwc8_dir']
    int8_cdhw32_dir = work_package['int8_cdhw32_dir']

    tgt_str = work_package['tgt_str']
    tgt_cases = work_package['tgt_cases']

    kits19tool.MULTI_POOL.starmap(preprocess_numpy_multiproc_helper,
                                  zip(tgt_cases,
                                      repeat(tgt_str),
                                      repeat(ref_dir),
                                      repeat(fp32_dir),

                                      repeat(fp16_dhwc8_dir),
                                      repeat(int8_cdhw32_dir),

                                      repeat(fp16_linear_dir),
                                      repeat(int8_linear_dir)
                                      ))


def preprocess_kits19_numpy(kits19tool: KITS19Tool) -> None:
    """
    Convert preprocessed KiTS19 reference input images into inference/calibration Numpy dataset
    Need to consider data format/type
    """
    ref_dir = kits19tool.PREPROCESSED_REF_DIR
    infer_dir = kits19tool.PREPROCESSED_INFER_DIR
    calib_dir = kits19tool.PREPROCESSED_CALIB_DIR
    infer_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    # populate files from preprocessed reference dataset
    print("Loading file names...")
    with open(Path(ref_dir, 'preprocessed_files.pkl'), "rb") as f:
        ref_pkl = pickle.load(f)

    ref_files = ref_pkl['file_list']
    infer_files = kits19tool.INFER_CASES
    calib_files = kits19tool.CALIB_CASES

    # save inference cases first and then calibration cases
    for tgt_str, tgt_cases, tgt_dir in [("infer", infer_files, infer_dir), ("calib", calib_files, calib_dir)]:

        # sanity check
        assert len(set(tgt_cases) - set(ref_files)) == 0,\
            "Preprocessed reference dataset: {} doesn't have all the {} cases: {}".format(
                ref_files, tgt_str, tgt_cases)

        fp32_dir = Path(tgt_dir, "fp32")
        fp16_linear_dir = Path(tgt_dir, "fp16")
        int8_linear_dir = Path(tgt_dir, "int8")

        fp16_dhwc8_dir = Path(tgt_dir, "fp16_dhwc8")
        int8_cdhw32_dir = Path(tgt_dir, "int8_cdhw32")

        fp32_dir.mkdir(parents=True, exist_ok=True)
        if tgt_str != 'calib':
            fp16_linear_dir.mkdir(parents=True, exist_ok=True)
            int8_linear_dir.mkdir(parents=True, exist_ok=True)

            fp16_dhwc8_dir.mkdir(parents=True, exist_ok=True)
            int8_cdhw32_dir.mkdir(parents=True, exist_ok=True)

        print("Converting data for {} dataset...".format(tgt_str))
        work_package = {
            'ref_dir': ref_dir,
            'fp32_dir': fp32_dir,
            'fp16_linear_dir': fp16_linear_dir,
            'int8_linear_dir': int8_linear_dir,

            'fp16_dhwc8_dir': fp16_dhwc8_dir,
            'int8_cdhw32_dir': int8_cdhw32_dir,

            'tgt_str': tgt_str,
            'tgt_cases': tgt_cases,
            'tgt_dir': tgt_dir,
        }
        preprocess_numpy_with_multiproc(kits19tool, work_package)


def gaussian_kernel(n: int, std: float) -> np.ndarray:
    """
    Returns gaussian kernel; std is standard deviation and n is number of points
    """
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D)
    gaussian3D = gaussian3D.reshape(n, n, n)
    gaussian3D = np.cbrt(gaussian3D)
    gaussian3D /= gaussian3D.max()
    return gaussian3D


def get_slice_for_sliding_window(image: np.ndarray, roi_shape: list, overlap: float
                                 ) -> Generator[list, None, None]:
    """
    Returns indices for image stride, to fulfill sliding window inference
    Stride is determined by roi_shape and overlap
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape),\
        f"Need proper ROI shape: {roi_shape}"
    assert isinstance(overlap, float) and overlap > 0 and overlap < 1,\
        f"Need sliding window overlap factor in (0,1): {overlap}"

    image_shape = image.shape
    dim = len(image_shape)
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

    size = [(image_shape[i] - roi_shape[i]) //
            strides[i] + 1 for i in range(dim)]
    i_range = range(0, strides[0] * size[0], strides[0])
    j_range = range(0, strides[1] * size[1], strides[1])
    k_range = range(0, strides[2] * size[2], strides[2])
    total_itr_left = len(i_range) * len(j_range) * len(k_range)
    for i in i_range:
        for j in j_range:
            for k in k_range:
                total_itr_left -= 1
                yield i, j, k, total_itr_left


def save_preconditioned_patches(kits19tool: KITS19Tool) -> None:
    """
    Store preconditioned gaussian patches used for sliding window
    """
    print("Saving preconditioned Gaussian window patches...")
    etc_dir = kits19tool.PREPROCESSED_ETC_DIR
    etc_dir.mkdir(parents=True, exist_ok=True)
    ROI_SHAPE = kits19tool.ROI_SHAPE
    SLIDE_OVERLAP_FACTOR = kits19tool.SLIDE_OVERLAP_FACTOR
    tgt_dtype = np.float16

    image_shape = [256, 256, 256]

    image = np.zeros(shape=image_shape, dtype=tgt_dtype)
    norm_map = np.zeros_like(image)
    norm_patch = gaussian_kernel(ROI_SHAPE[0], 0.125 * ROI_SHAPE[0]).astype(tgt_dtype)

    norm_patches = dict()

    for i, j, k, l in get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
        norm_map_slice = norm_map[
            ...,
            i:(ROI_SHAPE[0] + i),
            j:(ROI_SHAPE[1] + j),
            k:(ROI_SHAPE[2] + k)]

        norm_map_slice += norm_patch

    # each dim is: 0 -- start corner, 1 -- middle, 2 -- end corner
    for i, j, k, l in get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
        my_id = [0 if i == 0 else 2 if i + ROI_SHAPE[0] == image_shape[0] else 1,
                 0 if j == 0 else 2 if j + ROI_SHAPE[1] == image_shape[1] else 1,
                 0 if k == 0 else 2 if k + ROI_SHAPE[2] == image_shape[2] else 1]
        patch_id = my_id[0] * 9 + my_id[1] * 3 + my_id[2]
        my_slice = norm_map[
            ...,
            i:(ROI_SHAPE[0] + i),
            j:(ROI_SHAPE[1] + j),
            k:(ROI_SHAPE[2] + k)]

        norm_patches[patch_id] = norm_patch / my_slice

    assert len(norm_patches.keys()) == 27, "Not all the required patches generated"

    norm_patch_list = list()
    for _k in sorted(norm_patches):
        norm_patch_list.append(norm_patches[_k])

    np.save(Path(etc_dir, "gaussian_patches.npy"), np.stack(norm_patch_list))


def parse_args() -> argparse.Namespace:
    """
    Args used for preprocessing RAW data and the converted Numpy data


    Returns:
        argparse.Namespace:
            Namespace populated with argument strings and associated attributes

        arguments:
            --data_dir / -d             : directory storing mlperf-inference RAW data; KiTS19 subdir under this will be used
            --preprocessed_data_dir / -o: directory storing mlperf-inference preprocessed data; KiTS19 subdir under this dir will be used
            --num_proc / -n             : number of processes to be used in processing data

    """
    PARSER = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    PARSER.add_argument('--data_dir', '-d',
                        dest='data_dir',
                        default='build/data',
                        help="Dir storing mlperf-inference RAW data: KiTS19 subdir under this will be used")
    PARSER.add_argument('--preprocessed_data_dir', '-o',
                        dest='preprocessed_data_dir',
                        default='build/preprocessed_data',
                        help="Dir storing mlperf-inference preprocessed data: KiTS19 subdir under this dir will be used")
    PARSER.add_argument('--num_proc', '-n',
                        dest='num_proc',
                        type=int,
                        choices=list(range(1, 17)),
                        default=4,
                        help="Number of processes to be used")

    args = PARSER.parse_args()

    return args


def main() -> None:
    """
    Runs preprocess, verify integrity or regenerate MD5 hashes
    """
    args = parse_args()
    kits19tool = KITS19Tool(args)

    # Preprocess KiTS19 RAW data
    preprocess_kits19_raw_data(kits19tool)

    # Convert preprocessed KiTS19 RAW data into Numpy data
    preprocess_kits19_numpy(kits19tool)

    # Store preconditioned gaussian patches for optimizing post-processing
    save_preconditioned_patches(kits19tool)

    # all done
    print("Done!")


if __name__ == '__main__':
    main()
