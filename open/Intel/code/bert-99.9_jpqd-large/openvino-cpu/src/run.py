# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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

# This file is identical to https://github.com/mlcommons/inference/blob/master/language/bert/run.py
# except modifications to arguments for OpenVINO, make OV backend available (only),
# change default log path, and remove running accuracy script.


import mlperf_loadgen as lg
import argparse
import logging
import os
import sys
sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=['openvino'], default="openvino", help="Backend")
    parser.add_argument("--scenario", type=str, choices=["Offline","Server"], default="Offline",
                        help="Scenario")
    parser.add_argument("--accuracy", action="store_true",
                        help="enable accuracy pass")
    parser.add_argument("--max_examples", type=int,
                        help="Maximum number of examples to consider (not limited by default)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="name of model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="path to model")
    parser.add_argument("--data_path", type=str, default=None,
                        help="path to data")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="directory to save logs")
    parser.add_argument("--debug", action="store_true", help="debug, turn traces on")

    # OPENVINO Specific
    parser.add_argument('--nireq', type=check_positive, required=False, default=0,
                      help='Optional. Number of infer requests. Default value is determined automatically for device.')
    parser.add_argument('--nstreams', type=str, required=False, default=None,
                      help='Optional. Number of streams to use for inference on the CPU/GPU/MYRIAD '
                           '(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> '
                           'or just <nstreams>). '
                           'Default value is determined automatically for a device. Please note that although the automatic selection '
                           'usually provides a reasonable performance, it still may be non - optimal for some cases, especially for very small networks. '
                           'Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency '
                           'estimations the number of streams should be set to 1. '
                           'See samples README for more details.')
    parser.add_argument('--nthreads', type=int, required=False, default=None,
                      help='Number of threads to use for inference on the CPU, GNA '
                           '(including HETERO and MULTI cases).')
    parser.add_argument('--inference_precision', type=str, default='bf16',
                        choices=['bf16', 'f32'],
                        help='Inference precision for OpenVINO')

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf_conf", type=str, default="mlperf.conf", help="mlperf rules config")
    # file for user LoadGen settings such as target QPS
    parser.add_argument("--user_conf", type=str, default="user.conf", help="user config for user LoadGen settings such as target QPS")
    # file for LoadGen audit settings
    parser.add_argument("--audit_conf", type=str, default="audit.config", help="config for LoadGen audit settings")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--performance-sample-count", type=int, help="performance sample count")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--samples-per-query", default=8, type=int, help="mlperf multi-stream samples per query")
    args = parser.parse_args()
    args.batch_size = 1
    return args


def main():
    args = get_args()

    log.info(args)

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing.
    count_override = False
    count = args.count
    if count:
        count_override = True

    if args.backend == "openvino":
        from openvino_SUT import get_openvino_sut
        sut = get_openvino_sut(args)
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))

    mlperf_conf = os.path.abspath(args.mlperf_conf)
    if not os.path.exists(mlperf_conf):
        log.error("{} not found".format(mlperf_conf))
        sys.exit(1)

    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
        sys.exit(1)

    audit_config = os.path.abspath(args.audit_conf)

    if args.log_dir:
        log_path = os.path.abspath(args.log_dir)
        os.makedirs(log_path, exist_ok=True)
        os.chdir(log_path)

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.enable_trace = args.debug
    log_settings.log_output = log_output_settings

    settings = lg.TestSettings()
    settings.FromConfig(mlperf_conf, args.model_name, args.scenario)
    settings.FromConfig(user_conf, args.model_name, args.scenario)
    settings.scenario = SCENARIO_MAP[args.scenario]

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC

    if count_override:
        settings.min_query_count = count
        settings.max_query_count = count

    if args.samples_per_query:
        settings.multi_stream_samples_per_query = args.samples_per_query

    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_expected_latency_ns = int(args.max_latency * NANO_SEC)


    print("Running LoadGen test...")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings, audit_config)

    print("Done!")

    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl.qsl)


if __name__ == "__main__":
    main()
