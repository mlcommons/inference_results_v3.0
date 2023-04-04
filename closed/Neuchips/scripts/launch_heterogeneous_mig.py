#! /usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

'''
This script facilitates running an individual MLPerf Inference benchmark on a GPU instance while ensuring that
other GPU instances on the same GPU are simultaneously running other MLPerf Inference benchmarks in the background.

Given a set of benchmarks to run in the background and a benchmark to run as the 'main benchmark', it will start
each backround benchmark on a different GPU instance and then monitor to ensure that all background benchmarks have
entered the timed phase of their execution. Once that is detected, it will launch the main benchmark.

Upon completion of the main benchmark, it checks to ensure that the background benchmarks are still running and
waits for them to finish. This allows the user to confirm that measurements for the main benchmark are taken while
all background benchmarks are still running.

Example commands:

- main benchmark: ssd-resnet34, background benchmarks: datacenter, Server scenario, PerformanceOnly run
    python3 ./scripts/launch_heterogeneous_mig.py --main_action=run_harness --background_benchmarks=datacenter
                                                  --main_benchmark=ssd-resnet34 --main_scenario=server

- main benchmark: ssd-resnet34, background benchmarks: datacenter, Server scenario, AccuracyOnly run
    python3 ./scripts/launch_heterogeneous_mig.py --main_action=run_harness --background_benchmarks=datacenter
                                                  --main_benchmark=ssd-resnet34 --main_scenario=server
                                                  --main_benchmark_runargs="--test_mode=AccuracyOnly"

- main benchmark: resnet50, background benchmarks: datacenter, Server scenario, AUDIT_TEST01 run
    python3 ./scripts/launch_heterogeneous_mig.py --main_action=run_audit_test01 --background_benchmarks=datacenter
                                                  --main_benchmark=resnet50 --main_scenario=Offline

'''

import sys
import os
sys.path.insert(0, os.getcwd())

import signal
import subprocess
import argparse
import psutil
import time
import datetime
import threading
import queue

from pathlib import Path
from typing import Dict, List, Tuple, Final, Set

from code.common import logging
from code.common.constants import Benchmark, Scenario, G_DATACENTER_BENCHMARKS, G_EDGE_BENCHMARKS, AliasedName
from code.common.systems.accelerator import MIG, MIG_INFO_SOURCE
from code.common.systems.systems import SystemConfiguration
from code.common.systems.system_list import DETECTED_SYSTEM, MATCHED_SYSTEM
from code.common.systems.known_hardware import KnownMIG

# logging format and default level
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s")

# Supported scenarios per benchmark; for all the mlperf-inference benchmarks
# Once code.common defines the benchmark/scenario relationship, this can be replaced with it
BenchmarkScenarioMapT = Dict[str, Set[str]]
BENCHMARK_SCENARIO_ALL: Final[BenchmarkScenarioMapT] = {
    Benchmark.UNET3D: {Scenario.Offline, Scenario.SingleStream},
    Benchmark.BERT: {Scenario.Offline, Scenario.SingleStream, Scenario.Server},
    Benchmark.DLRM: {Scenario.Offline, Scenario.Server},
    Benchmark.RNNT: {Scenario.Offline, Scenario.SingleStream, Scenario.Server},
    Benchmark.ResNet50: {Scenario.Offline, Scenario.SingleStream, Scenario.MultiStream, Scenario.Server},
    Benchmark.SSDMobileNet: {Scenario.Offline, Scenario.SingleStream, Scenario.MultiStream},
    Benchmark.SSDResNet34: {Scenario.Offline, Scenario.SingleStream, Scenario.Server, Scenario.MultiStream},
}

# A100 SXM4 80GB 1g.10gb and A30 1g.6gb supports all the benchmarks/scenarios
# All other cases are treated as unknown
SUPPORT_MATRIX: Final[Dict[AliasedName, BenchmarkScenarioMapT]] = {
    KnownMIG.A100_SXM_80GB_1GPC.value.name: BENCHMARK_SCENARIO_ALL,
    KnownMIG.A100_PCIe_80GB_1GPC.value.name: BENCHMARK_SCENARIO_ALL,
    KnownMIG.A30_1GPC.value.name: BENCHMARK_SCENARIO_ALL,
}

# Default config for each benchmark
BENCHMARK_DEFAULT_CONFIG: Final[Dict[Benchmark, str]] = {
    Benchmark.UNET3D: 'hetero_high_accuracy',
    Benchmark.BERT: 'hetero',
    Benchmark.DLRM: 'hetero_high_accuracy',
    Benchmark.RNNT: 'hetero',
    Benchmark.ResNet50: 'hetero',
    Benchmark.SSDMobileNet: 'hetero',
    Benchmark.SSDResNet34: 'hetero',
}


def get_args() -> argparse.Namespace:
    """
    Args used for running heterogeneous benchmarks in homogeneous MIG setup.

    Returns:
        argparse.Namespace:
            Namespace populated with argument strings and associated attributes

        Some important arguments:
            --main_action          : specify what kind of run it should be, i.e. run_harness or run_audit_test01
            --main_benchmark       : main benchmark to measure performance with LoadGen log
            --main_scenario        : which scenario main benchmark is run for, i.e. offline or server
            --background_benchmarks: submission category, i.e. datacenter, from which background benchmarks to choose from
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--main_benchmark",
        default='resnet50',
        type=str.lower,
        help="The main benchmark from which to measure performance",
    )
    parser.add_argument(
        "--main_scenario",
        help="The scenario to run for the main benchmark",
        default='offline',
        type=str.lower,
        choices=['singlestream', 'multistream', 'offline', 'server'],
    )
    parser.add_argument(
        "--main_action",
        default="run_harness",
        choices=["run_harness", "run_audit_test01", "run_audit_test04", "run_audit_test05"],
        help="Makefile target for the main benchmark.",
    )
    parser.add_argument(
        "--main_benchmark_duration",
        type=int,
        default=600000,
        help="Duration for which to run the main benchmark in milliseconds",
    )
    parser.add_argument(
        "--main_benchmark_runargs",
        type=str,
        default="",
        help=("Additional arguments to be passed in the harness RUN_ARGS for the main benchmark, "
              "passed as a string in quotes"),
    )
    parser.add_argument(
        "--main_benchmark_cmd_prefix",
        type=str,
        default="",
        help=("Cmd to prepend for the main benchmark, passed as a string in quotes "
              "(Useful for profiling, i.e. 'nsys profile ...')"),
    )
    parser.add_argument(
        "--start_time_buffer",
        type=int,
        default=600000,
        help=("Time to delay between launching the background workloads and launching the main benchmark "
              "(whether or not all background benchmarks have started) in milliseconds"),
    )
    parser.add_argument(
        "--main_benchmark_immediate_start",
        type=bool,
        default=True,
        help="If all the background benchmarks have started, main benchmark starts immediately"
    )
    parser.add_argument(
        "--end_time_buffer_min",
        type=int,
        default=30000,
        help=("Minimum acceptable time between completion of the main benchmark and "
              "completion of the background benchmarks in milliseconds"),
    )
    parser.add_argument(
        "--background_benchmarks",
        type=str,
        default='all',
        help=("The set of benchmarks to run in the background as a csv with no spaces "
              "(macros: 'all', 'none', 'edge', and 'datacenter' are also supported)"),
    )
    parser.add_argument(
        "--background_benchmark_duration",
        type=str,
        default='automatic',
        help=("Duration for which to run the background benchmarks in milliseconds. If not specified, "
              "script automatically maintains background benchmarks until the main benchmark finishes."),
    )
    parser.add_argument(
        "--background_benchmark_timeout",
        type=int,
        default=7200000,
        help=("Timeout of the background benchmarks if duration == automatic, in milliseconds. "
              "Default to 120 minutes."),
    )
    parser.add_argument(
        "--lenient",
        action='store_true',
        help="Allow the run to proceed even if desired overlap of benchmarks is not achieved"
    )
    parser.add_argument(
        "--dryrun",
        action='store_true',
        help="Just print the benchmark commands that would be run",
    )
    parser.add_argument(
        "--verbose_all",
        action='store_true',
        help="Verbose logging from all the benchmarks, including background benchmarks",
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Verbose logging from the main benchmark",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Change logging level to DEBUG",
    )
    return parser.parse_args()


def get_mig_identification(system: SystemConfiguration) -> str:
    """
    Returns string of GPU & subsequent MIG instance identification.
    Also checks if MIG is enabled and the MIG configuration is homogeneous.
    Return string omits how many MIG instances are populated.

    Args:
        system (SystemConfiguration):
            The system information on the GPUs and their configuration.
            The information is compliant to one of the mlperf-inference submission systems.

    Returns:
        str:
            A string showing MIG name that's identifiable through KnownMIG, i.e. registered
            MIG configuration name compliant to one of the mlperf-inference submission systems.
            ex) A100-SXM-80GB MIG-1g.10gb
    """
    assert system.accelerator_conf.num_migs() > 0, "System has no detected MIG slices"
    accelerators = system.accelerator_conf.get_accelerators()
    # Check for multiple types of MIGs
    mig = None
    for accelerator in accelerators:
        if accelerator.__class__ is not MIG:
            continue
        elif mig is None:
            mig = accelerator
        else:
            raise Exception("MIG config is not homogenous.")
    # Reformat the MIG slice name to be something that the hetero-mig script understands.
    return str(mig.name)


def get_target_mig_uuids(system: SystemConfiguration) -> List[str]:
    """
    Returns collection of unique MIG UUIDs used for hetero MIG benchmark run.
    In case of MIG config in multi-GPU system, the target MIG instances are chosen
    from the GPU in which the largest number of MIG instances are found.
    However, under the homogeneous MIG config assumption, which GPU is selected may
    not relevant. The above is to make sure the safest selection guranteed.

    Args:
        system (SystemConfiguration):
            The system information on the GPUs and their configuration.
            The information is compliant to one of the mlperf-inference submission systems.

    Returns:
        List[str]:
            list of str: MIG_UUID
            ex) [MIG-1e8534f8-5dd9-575e-bcee-82f3db1bebd1,
                 MIG-818abda7-8ede-58fa-bb02-ed3d388b6922]
    """
    mig_uuids = dict()
    for mig_slice in MIG_INFO_SOURCE:
        if mig_slice["parent_gpu_uuid"] not in mig_uuids:
            mig_uuids[mig_slice["parent_gpu_uuid"]] = []
        mig_uuids[mig_slice["parent_gpu_uuid"]].append(mig_slice["uuid"])

    # Get the GPU with most number of MIG instances
    m = max(mig_uuids, key=lambda _k: len(mig_uuids[_k]))
    return mig_uuids[m]


def enqueue_output(pipe_out: subprocess.PIPE, my_q: queue.Queue) -> None:
    """
    Stores printouts from stream.
    Used for collecting printouts from background benchmarks.

    NOTE: This callable will eventually close out PIPE as the subprocess associated with terminates.

    Args:
        pipe_out (subprocess.PIPE):
            stream of standard out from processed opened for background benchmark run.

        my_q (queue.Queue):
            FIFO queue that holds bitstream captured from pipe_out.
    """
    for line in iter(pipe_out.readline, b''):
        line = line.strip()
        if line:
            my_q.put(line)
        time.sleep(.5)
    pipe_out.close()


def kill_process(process: subprocess.Popen) -> None:
    """
    Kills the process group.
    Usually called for terminating a specific background process.

    Args:
        process (subprocess.Popen):
            process handle opened for background benchmark runs.
    """
    try:
        for child in psutil.Process(process.pid).children(recursive=True):
            child.kill()
        process.kill()
    except psutil.NoSuchProcess:
        pass


def early_exit(processes: List[subprocess.Popen], lenient: bool) -> bool:
    """
    Kills the process group.
    Usually called for terminating all known background process under unexpected circumstances.

    NOTE: Calling this function will send an intention to main process to fail.

    Args:
        process (List[subprocess.Popen]):
            list of process handles opened for background benchmark runs.
        lenient (bool):
            If True, do not terminate background process.

    Returns:
        bool:
            Acknowledges current program that subprocesses exited early, and guides to exit as well
    """
    print("Early exiting, killing all the background processes...")
    if not lenient:
        for process in processes:
            kill_process(process)
        return True
    return False


def get_background_benchmarks(GPU: AliasedName,
                              num_mig_instances: int,
                              main_benchmark: str,
                              background_benchmark_input: str,
                              background_duration: str,
                              background_timeout: int) -> Tuple[List[str], int, str]:
    """
    Returns information of background benchmarks, i.e. list of benchmarks selected
    to run in the background, duration of those runs, and the action commandline argument
    for those runs.
    Main benchmark is removed from background benchmarks to avoid any possible interference.

    Args:
        GPU (AliasedName):
            AliasedName representing the MIG instance matched with one of the KnownMIG.
        num_mig_instances (int):
            Number of MIGs enabled in the GPU
        main_benchmark (str):
            Name of the main benchmark to run, as in mlperf-inference.
        background_benchmark_input (str):
            Comma separated list of benchmarks to be run in background.
            Also can be specified for the category to choose background benchmarks from.
            See also --background_benchmarks from commandline arguments.
        background_duration (str):
            Time to run the background benchmarks. Note that this is string type.
            If 'automatic' is provided, algorithm determines the run duration.
            If string with digits is provided, it is translated into time in milliseconds.
        background_timeout (int):
            Time to wait before auto-terminating background benchmarks, in milliseconds.

    Returns:
        Tuple[List[str], int, str]:
            list of str:
                background_benchmarks -- benchmark names to be used for background runs.
                ex) ["DLRM", "ResNet50", "SSD-ResNet34"]
            int:
                background_benchmark_duration -- runtime in milliseconds for background runs.
            str:
                background_benchmark_action -- commandline argument to be used for background runs.
    """
    # to return, default value
    background_benchmarks = list()
    background_benchmark_duration = 0
    background_benchmark_action = "run_harness"

    from_benchmark_group = background_benchmark_input in {'all', 'datacenter', 'edge'}

    # Expand the list of background benchmarks if macro was used and
    # then validate that the selected benchmarks are all supported.
    if from_benchmark_group:
        background_benchmarks = list({
            'datacenter': G_DATACENTER_BENCHMARKS,
            'edge': G_EDGE_BENCHMARKS,
            'all': list(Benchmark),
        }[background_benchmark_input])
        background_benchmarks = [b.valstr() for b in background_benchmarks]
    elif background_benchmark_input == 'none':
        background_benchmarks = list()
    else:
        background_benchmarks = background_benchmark_input.split(',')
        valid_benchmarks = [_b in list(Benchmark) for _b in background_benchmarks]
        assert all(valid_benchmarks) and any(background_benchmarks),\
            "Unexpected background benchmark(s): {}".format(background_benchmark_input)

    # Just for determinism :)
    background_benchmarks.sort()
    for benchmark in background_benchmarks:
        assert Benchmark.get_match(benchmark) in SUPPORT_MATRIX[GPU],\
            "Specified background benchmark {} is not supported".format(benchmark)

    # Set the duration for which to run the background benchmark, if user provided any
    if background_duration.isdigit():
        background_benchmark_duration = int(background_duration)
    # If the background_benchmark_duration is automatic, let background benchmarks run until main benchmark ends.
    # The timeout would be 60 minutes for now.
    elif background_duration.lower() == 'automatic':
        logging.info(("Setting the background benchmark to run for indefinitely long "
                      "until main benchmark finishes."))
        background_benchmark_duration = background_timeout
    else:
        assert false, "Unknown setting for background_benchmark_duration: {}".format(background_duration)

    # If there are enough benchmarks to run in background, remove main benchmark
    if len(background_benchmarks) >= num_mig_instances and\
       from_benchmark_group and\
       main_benchmark in background_benchmarks:
        background_benchmarks.remove(main_benchmark)

    # If we have more background_benchmarks than number of MIGs, choose from sorted name for reproducibility
    if len(background_benchmarks) >= num_mig_instances:
        background_benchmarks = background_benchmarks[:(num_mig_instances - 1)]

    return background_benchmarks, background_benchmark_duration, background_benchmark_action


def get_cmd_templates(main_benchmark_runargs: str,
                      main_benchmark_prefix: str,
                      verbose: bool, verbose_all: bool,
                      background_logdir: str) -> Tuple[str, str]:
    """
    Returns command templates to run the main and background benchmarks.
    These command templates

    Args:
        main_benchmark_runargs (str):
            Run argument(s) to be added for main benchmark run.
        main_benchmark_prefix (str):
            Cmd to prepend for the main benchmark.
        verbose (bool):
            Use verbose logging for main benchmark.
        verbose_all (bool):
            Use verbose logging for all the benchmarks, including main and background benchmarks.
        background_logdir (str):
            Points to the directory where background LoadGen logs are going to be kept.

    Returns:
        Tuple[str, str]:
            str:
                Commandline string used as template for main benchmark.
            str:
                Commandline string used as template for background benchmark(s).
    """
    # Template command for running each of the benchmarks
    cuda_prefix = ['CUDA_VISIBLE_DEVICES={}']
    cmd_template_prefix = \
        ['make', '{}', 'RUN_ARGS="--benchmarks={}',
         '--scenarios={}', '--config_ver={}', '--min_duration={}']
    cmd_template_suffix = ['"']

    # Always set the min_query_count to 1 for background benchmarks so that their duration
    # is purely based on the min_duration setting
    background_runargs = ['--min_query_count=1']
    background_extra = [f'LOG_DIR={background_logdir}']
    if verbose_all:
        background_runargs += ['--verbose', '--verbose_nvtx']

    main_prefix = main_benchmark_prefix.split()

    main_runargs = main_benchmark_runargs.split()
    if verbose_all or verbose:
        main_runargs += ['--verbose', '--verbose_nvtx']

    background_cmd_template = " ".join(cuda_prefix + cmd_template_prefix + background_runargs +
                                       cmd_template_suffix + background_extra)
    main_cmd_template = " ".join(cuda_prefix + main_prefix +
                                 cmd_template_prefix + main_runargs + cmd_template_suffix)

    return main_cmd_template, background_cmd_template


def launch_background_benchmarks(background_benchmarks: List[str],
                                 background_cmd_template: str,
                                 background_benchmark_duration: int,
                                 background_benchmark_action: str,
                                 start_time_buffer: int,
                                 main_benchmark_immediate_start: bool,
                                 mig_uuids: List[str],
                                 lenient: bool,
                                 dryrun: bool) -> Tuple[List[subprocess.Popen], List[bool], str]:
    """
    Launches background benchmarks and waits until they start inferences.

    Args:
        background_benchmarks (List[str]):
            background_benchmarks -- benchmark names to be used for background runs
            ex) ["DLRM", "ResNet50", "SSD-ResNet34"]
        background_cmd_template (str):
            Commandline string used as template for background benchmark(s)
        background_benchmark_duration (int):
            Time to run the background benchmarks, in milliseconds.
        background_benchmark_action (str):
            Commandline argument to be used for background runs.
        start_time_buffer (int):
            Gap between launching the background benchmark and main benchmark, in milliseconds.
        main_benchmark_immediate_start (bool):
            Start main benchmark immediately, as soon as background benchmarks start inferences.
            See also --main_benchmark_immediate_start from commandline arguments.
        mig_uuids (List[str]):
            List containing MIG_UUIDs that will be used for benchmark runs.
            First one is reserved for main benchmark run.
            ex) [MIG-1e8534f8-5dd9-575e-bcee-82f3db1bebd1,
                 MIG-818abda7-8ede-58fa-bb02-ed3d388b6922]
        lenient (bool):
            Whether to terminate run on any background benchmark anomaly.
            See also --lenient from commandline arguments.
        dryrun (bool):
            Just generate cmds, instead of really run benchmarks.
            See also --dryrun from commandline arguments.

    Returns:
        List[subprocess.Popen], List[bool], str]:
            List[subprocess.Popen]:
                background_processes -- holds handles of background benchmark processes launched.
            List[bool]:
                completed_benchmarks -- holds flags of background benchmarks that are completed already.
            str:
                debug_msg -- holds string containing debug information
    """
    # to return
    background_processes = list()
    completed_benchmarks = [False, ] * len(background_benchmarks)
    debug_msg = ""

    if background_benchmarks:
        # Set logging level if something is wrong
        logging_wrong = logging.warn if lenient else logging.error

        for i, background_benchmark in enumerate(background_benchmarks):
            # scenario is fixed as Offline for background benchmarks
            cmd = background_cmd_template.format(
                mig_uuids[i + 1],
                background_benchmark_action,
                background_benchmark,
                'offline',
                BENCHMARK_DEFAULT_CONFIG[Benchmark.get_match(background_benchmark)],
                background_benchmark_duration
            )
            if dryrun:
                debug_msg += cmd + '\n'
            else:
                logging.info('Launching background workload: {}'.format(cmd))
                background_processes.append(subprocess.Popen(cmd, universal_newlines=True, shell=True,
                                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT))

        if not dryrun:
            # Checking explicitly to detect that benchmarks have started by monitoring their logs in real-time
            # Using mechanism described on:
            # https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
            # to poll stdout of each background benchmark
            await_start_timeout = start_time_buffer / 1000
            logging.info(("Waiting for background benchmarks to reach timed section "
                          "with a {} second timeout".format(await_start_timeout)))
            background_launch_time = datetime.datetime.now()
            background_queues = [queue.Queue() for _ in background_benchmarks]
            assert len(background_queues) == len(background_processes), "Error in prep"
            background_threads = [threading.Thread(target=enqueue_output,
                                                   args=(background_processes[i].stdout, myq))
                                  for i, myq in enumerate(background_queues)]
            for t in background_threads:
                t.daemon = True
                t.start()
            unstarted_benchmarks = [True, ] * (len(background_benchmarks))
            termination_check_counter = 0
            while ((datetime.datetime.now() - background_launch_time).seconds < await_start_timeout) and\
                    (unstarted_benchmarks.count(True) > 0):
                found_lines = False
                for i, benchmark in enumerate(background_benchmarks):
                    if unstarted_benchmarks[i]:
                        try:
                            line = background_queues[i].get_nowait()
                        except queue.Empty:
                            pass
                        else:
                            found_lines = True
                            line = line.strip()
                            logging.debug("[{} {}]:".format(i, benchmark).upper() + line)
                            if line.endswith("running actual test."):
                                logging.info("Detected that background benchmark {}, {} has started".format(
                                    i, background_benchmarks[i]))
                                unstarted_benchmarks[i] = False
                if not found_lines:
                    termination_check_counter += 1
                    if termination_check_counter == 30:
                        for i, benchmark in enumerate(background_benchmarks):
                            process_poll_result = background_processes[i].poll()
                            if not (process_poll_result is None):
                                unstarted_benchmarks[i] = False
                                completed_benchmarks[i] = True
                                logging_wrong(
                                    "Background benchmark {}, {} exited earlier than expected with code {}".format(
                                        i, background_benchmarks[i], process_poll_result)
                                )
                                to_early_exit = early_exit(background_processes, lenient)
                                if to_early_exit:
                                    debug_msg += "CRITICAL: EARLY EXIT\n"
                                    return [], [], debug_msg
                    time.sleep(.1)

            early_starters = []
            if unstarted_benchmarks.count(True) > 0:
                early_starters = ["({}, {})".format(i, background_benchmarks[i])
                                  for i in range(len(background_benchmarks)) if unstarted_benchmarks[i]]
                logging_wrong(("Did not detect start of background benchmarks {} "
                               "after waiting for specified delay.".format(early_starters)))
                to_early_exit = early_exit(background_processes, lenient)
                if to_early_exit:
                    debug_msg += "CRITICAL: EARLY EXIT\n"
                    return [], [], debug_msg

            logging.info("All background benchmarks have started")
            remaining_delay = await_start_timeout - (datetime.datetime.now() - background_launch_time).seconds
            if not main_benchmark_immediate_start and remaining_delay > 0:
                logging.info("Waiting for remaining delay of {} seconds.".format(remaining_delay))
                time.sleep(remaining_delay)

    return background_processes, completed_benchmarks, debug_msg


def run_main_benchmark(main_benchmark: str,
                       main_scenario: str,
                       main_action: str,
                       main_benchmark_duration: int,
                       main_cmd_template: str,
                       background_check_auto: bool,
                       background_benchmarks: List[str],
                       background_processes: List[subprocess.Popen],
                       completed_benchmarks: List[bool],
                       mig_uuids: List[str],
                       lenient: bool,
                       dryrun: bool) -> str:
    """
    Launches background benchmarks and waits until they start inferences.

    Args:
        main_benchmarks (str):
            Benchmark name for main benchmark.
            See also --main_benchmark from commandline arguments.
            ex) "ResNet50"
        main_scenario (str):
            Scenario main benchmark to run for.
            See also --main_scenario from commandline arguments.
            ex) "Server"
        main_action (str):
            Commandline argument to be used for main benchmark run.
            See also --main_action from commandline arguments.
            ex) "run_harness"
        main_benchmark_duration (int):
            Time to run the main benchmark, in milliseconds.
            See also --main_benchmark_duration from commandline arguments.
        main_cmd_template (str):
            Commandline string used as template for main benchmark
        background_check_auto (bool):
            Manages the background benchmark progress automatically
        background_benchmarks (List[str]):
            Benchmark names to be used for background runs
            ex) ["DLRM", "ResNet50", "SSD-ResNet34"]
        background_processes (List[subprocess.Popen]):
            List that holds handles of background benchmark processes launched.
        completed_benchmarks (List[bool]):
            List that holds flags of background benchmarks, to track their progress.
        mig_uuids (List[str]):
            List containing MIG_UUIDs that will be used for benchmark runs.
            First one is reserved for main benchmark run.
            ex) [MIG-1e8534f8-5dd9-575e-bcee-82f3db1bebd1,
                 MIG-818abda7-8ede-58fa-bb02-ed3d388b6922]
        lenient (bool):
            Whether to terminate run on any background benchmark anomaly.
            See also --lenient from commandline arguments.
        dryrun (bool):
            Just generate cmd, instead of really run benchmark.
            See also --dryrun from commandline arguments.

    Returns:
        str:
            debug_msg -- holds string containing debug information
    """
    # Set logging level if something is wrong
    logging_wrong = logging.warn if lenient else logging.error

    # build the cmdline for main benchmark
    cmd = main_cmd_template.format(
        mig_uuids[0],
        main_action,
        main_benchmark,
        main_scenario,
        BENCHMARK_DEFAULT_CONFIG[Benchmark.get_match(main_benchmark)],
        main_benchmark_duration
    )

    # to return
    debug_msg = ""

    # Main benchmark measurement, with housekeeping on background benchmarks
    if dryrun:
        debug_msg += cmd + '\n'
    else:
        logging.info("Launching main benchmark: {}".format(cmd))
        main_benchmark_process = subprocess.Popen(cmd, universal_newlines=True, shell=True,
                                                  stdout=sys.stdout, stderr=sys.stderr)
        main_exit_code = main_benchmark_process.wait()
        main_benchmark_complete_time = datetime.datetime.now()
        early_completions = completed_benchmarks.count(True)
        if main_exit_code != 0:
            logging_wrong("Main benchmark ended with non-zero exit code {}.".format(main_exit_code))
            to_early_exit = early_exit(background_processes, lenient)
            if to_early_exit:
                debug_msg += "CRITICAL: EARLY EXIT\n"
                return debug_msg
        logging.info("Main benchmark ended with exit code {}.".format(main_exit_code))

        if background_benchmarks:
            logging.info("Waiting for {} of {} background benchmarks to complete".format(
                len(background_benchmarks) - early_completions, len(background_benchmarks)))

            if background_check_auto:
                # Check if any of the background benchmarks exits with non-zero status
                # and stop all the background processes.
                for i, benchmark in enumerate(background_benchmarks):
                    if not completed_benchmarks[i]:
                        poll_result = background_processes[i].poll()
                        if not (poll_result is None):
                            completed_benchmarks[i] = True
                            if poll_result != 0:
                                logging_wrong("Background benchmark {} completed with non-zero exit code {}.".format(
                                    benchmark, poll_result))
                                to_early_exit = early_exit(background_processes, lenient)
                                if to_early_exit:
                                    debug_msg += "CRITICAL: EARLY EXIT\n"
                                    return debug_msg
                # Terminate all the processes
                logging.info("Automatically terminating all background benchmarks...")
                for i, process in enumerate(background_processes):
                    logging.info("Terminating background process {}:{}.".format(i, process.pid, background_benchmarks[i]))
                    kill_process(background_processes[i])
                # to exit
                return debug_msg
            else:
                while completed_benchmarks.count(True) < len(background_benchmarks):
                    for i, benchmark in enumerate(background_benchmarks):
                        if not completed_benchmarks[i]:
                            poll_result = background_processes[i].poll()
                            if not (poll_result is None):
                                # Don't want any background benchmark to exit before main benchmark
                                if (datetime.datetime.now() - main_benchmark_complete_time).seconds < args.end_time_buffer_min / 1000:
                                    early_completions += 1
                                    logging_wrong("Background benchmark {} completed too early.".format(benchmark))
                                completed_benchmarks[i] = True
                                # Background benchmark might have exited badly
                                if poll_result != 0:
                                    logging_wrong("Background benchmark {} completed with non-zero exit code {}.".format(
                                        benchmark, poll_result))
                                    to_early_exit = early_exit(background_processes, lenient)
                                    if to_early_exit:
                                        debug_msg += "CRITICAL: EARLY EXIT\n"
                                        return debug_msg
                                # Background benchmark behaved well
                                logging.info("Background benchmark {}, {} completed with exit code {}.".format(
                                    i, benchmark, poll_result))

            # Drain the stdout for all background benchmarks
            for i, benchmark in enumerate(background_benchmarks):
                with background_queues[i].mutex:
                    remaining_lines = list(background_queues[i].queue)
                    for line in remaining_lines:
                        line = line.strip()
                        logging.debug("[{}, {}]:".format(i, benchmark).upper() + line)

            if len(early_starters) == 0:
                logging.info("All background benchmarks started before the benchmark under test")
            else:
                logging_wrong(
                    ("Background benchmarks {} did not start long enough before the benchmark under test. "
                     "Consider a longer start_time_buffer setting".format(early_starters))
                )

            if early_completions == 0:
                logging.info("All background benchmarks completed after the benchmark under test")
            else:
                logging_wrong("{} background benchmarks completed too early".format(early_completions))

    return debug_msg


def main():
    """
    Runs heterogeneous benchmarks in homogeneous MIG setup.
    That is, running different benchmarks in the same kind of MIG instances,
    and measure one specific benchmark performance.
    """
    # Holds msg(s) helping debug/testing
    debug_msgs = ""

    # Parse cmdline arguments
    args = get_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    assert MATCHED_SYSTEM is not None, f"Unsupported system detected: {DETECTED_SYSTEM}."
    assert DETECTED_SYSTEM.accelerator_conf.num_migs() > 0, "System has no detected MIG slices"
    system = DETECTED_SYSTEM
    GPU_INSTANCE = get_mig_identification(system)
    SUPPORTED_GPU = None
    for k in SUPPORT_MATRIX:
        if k == GPU_INSTANCE:
            SUPPORTED_GPU = k
            break
    assert SUPPORTED_GPU, "Unsupported GPU_INSTANCE detected {}".format(GPU_INSTANCE)

    # Ensure that the scenario chosen for the main benchmark is supported
    assert Scenario.get_match(args.main_scenario) in SUPPORT_MATRIX[SUPPORTED_GPU][Benchmark.get_match(args.main_benchmark)],\
        "Specified main benchmark {} does not support the {} scenario".format(args.main_benchmark, args.main_scenario)

    # Get the value to select each GPU instance with CUDA_VISIBLE_DEVICES
    mig_uuids = get_target_mig_uuids(system)
    num_mig_instances = len(mig_uuids)
    assert num_mig_instances > 1, "Need multiple MIG instances: {}".format(mig_uuids)

    # Set up background benchmarks
    background_benchmarks, background_benchmark_duration, background_benchmark_action =\
        get_background_benchmarks(SUPPORTED_GPU,
                                  num_mig_instances,
                                  args.main_benchmark,
                                  args.background_benchmarks,
                                  args.background_benchmark_duration,
                                  args.background_benchmark_timeout)

    # Sanity check on populated background benchmarks
    # Some MIGs may be left idle, but at least one MIG should be able to run the main benchmark
    assert len(background_benchmarks) < num_mig_instances,\
        ("Should have fewer background benchmarks than MIG instances to leave one for the main benchmark: "
         "main benchmark: {}, background benchmarks: {}, # MIGs: {}".format(
             args.main_benchmark, background_benchmarks, num_mig_instances))

    # Create directory to store logs from the background benchmarks so they do not contaminate
    # build/logs for submission
    background_logdir_path = Path('build', 'MIG_hetero_background_logs')
    background_logdir_path.mkdir(parents=True, exist_ok=True)

    # get command templates for main and background runs
    main_cmd_template, background_cmd_template =\
        get_cmd_templates(args.main_benchmark_runargs, args.main_benchmark_cmd_prefix,
                          args.verbose, args.verbose_all, str(background_logdir_path))

    # Launch each of the background workloads
    background_processes, completed_benchmarks, debug_msg =\
        launch_background_benchmarks(background_benchmarks,
                                     background_cmd_template,
                                     background_benchmark_duration,
                                     background_benchmark_action,
                                     args.start_time_buffer,
                                     args.main_benchmark_immediate_start,
                                     mig_uuids,
                                     args.lenient,
                                     args.dryrun)
    debug_msgs += debug_msg
    assert "CRITICAL:" not in debug_msgs, "Critical error found {}".format(debug_msgs)

    # now run main benchmark
    debug_msg = run_main_benchmark(args.main_benchmark,
                                   args.main_scenario,
                                   args.main_action,
                                   args.main_benchmark_duration,
                                   main_cmd_template,
                                   args.background_benchmark_duration == "automatic",
                                   background_benchmarks,
                                   background_processes,
                                   completed_benchmarks,
                                   mig_uuids,
                                   args.lenient,
                                   args.dryrun)
    debug_msgs += debug_msg
    assert "CRITICAL:" not in debug_msgs, "Critical error found {}".format(debug_msgs)

    if debug_msgs:
        logging.info("Collected below msg(s) while running: \n{}".format(debug_msgs))


if __name__ == '__main__':
    PID = os.getpid()
    try:
        main()
    # to make sure interrupt kills all the processes
    except KeyboardInterrupt:
        PGID = os.getpgid(PID)
        os.killpg(PGID, signal.SIGKILL)
