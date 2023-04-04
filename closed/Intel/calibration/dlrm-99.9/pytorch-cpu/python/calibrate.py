"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import sys 
import torch
from shutil import copyfile
import torch_ipex as ipex

try:
    dlrm_dir_path = os.environ['DLRM_DIR']
    sys.path.append(dlrm_dir_path)
except KeyError:
    print("ERROR: Please set DLRM_DIR environment variable to the dlrm code location")
    sys.exit(0)

# the datasets we support
DATASETS_KEYS = ["kaggle", "terabyte"]

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "terabyte",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
    "dlrm-kaggle-pytorch": {
        "dataset": "kaggle",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 128,
    },
    "dlrm-terabyte-pytorch": {
        "dataset": "terabyte",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
}

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="name of the mlperf model, ie. dlrm")
    parser.add_argument("--model-path", required=True, help="path to the model file")
    parser.add_argument("--dataset", choices=DATASETS_KEYS, help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    parser.add_argument("--test-num-workers", type=int, default=0, help='# of workers reading the data')
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--max-batchsize", type=int, help="max batch size in a single inference")
    parser.add_argument("--output", help="test results")
    parser.add_argument("--inputs", help="model inputs (currently not used)")
    parser.add_argument("--outputs", help="model outputs (currently not used)")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument('--int8-configuration-dir', default='int8_configure.json', type=str, metavar='PATH',
                            help = 'path to int8 configures, default file name is configure.json')
    parser.add_argument("--calibration-batches", default=8, type=int, help="calibration-batches")

    args = parser.parse_args()
    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")

    return args

def get_dataset(args):
    from criteocalib import CriteoCalib
    kwargs = {"randomize": 'total',  "memory_map": True}
    # dataset to use
    # --count-samples can be used to limit the number of samples used for testing
    ds = CriteoCalib(data_path=args.dataset_path,
                        name=args.dataset,
                        test_num_workers=16,
                        max_ind_range=args.max_ind_range,
                        mlperf_bin_loader=args.mlperf_bin_loader,
                        sub_sample_rate=args.data_sub_sample_rate,
                        **kwargs)
    return ds


def input_wrap(X, lS_o, lS_i, device):
    lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) else lS_i.to(device)
    lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) else lS_o.to(device)
    X = X.to(device)
    return X, lS_o, lS_i

def calibration(model, data_loader, num_calib_batches, int8_configure_dir, device):
    conf = ipex.AmpConf(torch.int8)
    with torch.no_grad():
        for j, (X, lS_o, lS_i, T) in enumerate(data_loader):
            with ipex.AutoMixPrecision(conf, running_mode="calibration"):
                model(*input_wrap(X, lS_o, lS_i, device))
            if j == num_calib_batches:
                break;
    print("Complete calibration with", j, "batches and save result to", int8_configure_dir)
    conf.save(int8_configure_dir)
    return

def main():
    from backend_pytorch_native import get_backend
    args = get_args()
    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    device = ipex.DEVICE
    ipex.core.enable_auto_dnnl()
    ipex.core.set_execution_mode(train=False)
    ipex.core.enable_jit_opt()
    ds = get_dataset(args)
    backend = get_backend(args.backend, args.dataset, ipex.DEVICE, args.max_ind_range,
                          args.data_sub_sample_rate, False, True,
                          False)
    model = backend.load(args.model_path, args.inputs, args.outputs)
    dataloader = ds.get_calibration_data_loader()
    calibration(model, dataloader, args.calibration_batches, args.int8_configuration_dir, device)
    print("Calibration Done !!")


if __name__ == "__main__":
    main()
