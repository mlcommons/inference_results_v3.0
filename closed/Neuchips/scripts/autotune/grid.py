#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


""" GridSearch-based Autotuning

An automated means of executing an MLPerf benchmark+scenario with a variety of parameters/configurations
and parsing the results of execution.
"""
import argparse
import copy
import glob
import hashlib  # For directory naming
import importlib  # For python parsing!
import inspect  # For python parsing!
import json
import os
from pprint import pprint
import re
import sys
import time
from typing import List

sys.path.insert(0, os.getcwd())
from library import CartesianProduct, Run, SearcherInterface
from code.common import run_command  # used in tee
from code.common.constants import Benchmark, Scenario, config_ver_to_workload_setting
from code.common.systems.system_list import DETECTED_SYSTEM, MATCHED_SYSTEM, SystemClassifications
from code.common.result_parser import from_loadgen_by_keys, scenario_loadgen_log_keys
from code.main import main, parse_main_args
from configs.configuration import ConfigRegistry

# For metadata and stats we record for each run (in addition to what the harness already records)
METAFILE = "autotuneMETAFILE.json"


class ExecStats:
    """Named tuple that helps us understand how much work each autotuning session is doing."""

    def __init__(self):
        self.builds = 0
        self.cached_builds = 0
        self.runs = 0
        self.cached_runs = 0

    def __str__(self):
        return f"builds={self.builds}, cached_builds={self.cached_builds}, runs={self.runs}, cached_runs={self.cached_runs}"


class PerfStats:
    # List the loadgen keys we want
    scenario_keys_set = {v for k, v in scenario_loadgen_log_keys.items()} | {"result_validity"}
    _extra_key_set = {
        "result_min_latency_ns",
        "result_max_latency_ns",
        "result_mean_latency_ns",
        "result_50.00_percentile_latency_ns",
        "result_90.00_percentile_latency_ns",
        "result_95.00_percentile_latency_ns",
        "result_97.00_percentile_latency_ns",
        "result_99.00_percentile_latency_ns",
        "result_99.90_percentile_latency_ns",
    }
    verbose_stat_key_set = scenario_keys_set | _extra_key_set

    def __getitem__(self, item):
        return self.data[item]

    def _from_file(self, directory):
        """ Populate self.data from contents of directory which contains:
        - METAFILE at top level
        - mlperf_log_summary.txt in a run/platform-specific subdirectory.
        """
        verbose = args.verbose_stats
        search_path = os.path.join(directory, "**/mlperf_log_detail.txt")
        paths = [name for name in glob.glob(search_path, recursive=True)]
        if not paths:
            raise RuntimeError(f"Could not find mlperf_log_detail.txt in: \n{directory}\nDid you mean to run with --noparse?")
        key_set = self.verbose_stat_key_set if verbose else self.scenario_keys_set
        result = from_loadgen_by_keys(os.path.dirname(paths[0]), key_set)
        assert len(result) > 0
        to_ret = {}
        to_ret.update(result)

        with open(os.path.join(directory, METAFILE), 'r') as f:
            extra_stats = json.load(f)['run_info']
        to_ret.update(extra_stats)
        self.data = to_ret

    def __init__(self, directory_or_data):
        if isinstance(directory_or_data, str):
            self._from_file(directory_or_data)
        elif isinstance(directory_or_data, dict):
            self.data = directory_or_data
        else:
            raise RuntimeError("Unexpected argument type in initializer of PerfStats")


class ConfigGrid:
    """ An iterable to enumerate through either
     - A cartesian product of configurations (Default)
     - User-provided configurations (as defined by META_search_callback)

    Actual iterator is referred to as a "cross-term", which is a single
    value in the cartesian product.
    The cross-term on its own isn't very useful, but can be fed to
    cross_to* methods for more useful items.
    """

    def __init__(self, bench, scen, config_dict, config_funcs=None):
        """ Construct a ConfigGrid

        Args:
            bench (str): The benchmark requested (fuzzy match behavior using Benchmark.get_match)
            scen (str): The scenario requested (fuzzy match behavior using Scenario.get_match)
            config_dict (Dict[str, List]): A config dictionary. Refer to 'Config Schema' in the README for format
            config_funcs (Dict[str, Callable]): A dictionary of META* functions. Refer to 'Config Schema' in the README for requirements.

        """
        if args.spoof_system_id:
            self.system_id = args.spoof_system_id
        else:
            self.system = DETECTED_SYSTEM
            self.system_id = self.system.get_id()

        self.benchmark = Benchmark.get_match(bench)
        if self.benchmark is None:
            raise RuntimeError(f"'{bench}' is not a valid benchmark name.")
        self.scenario = Scenario.get_match(scen)
        if self.scenario is None:
            raise RuntimeError(f"'{scen}' is not a valid scenario name.")

        self.workload_setting = config_ver_to_workload_setting(self.benchmark, 'default')
        ConfigRegistry.load_configs(self.benchmark, self.scenario)
        self.base_config = ConfigRegistry.get(self.benchmark, self.scenario, self.system, **self.workload_setting.as_dict())
        if self.base_config is None:
            raise RuntimeError(f"Can't find config corresponding to {self.benchmark}, {self.scenario}, {self.system_id}")

        griddict = config_dict
        self.no_rebuild_params = None
        # No-op
        self.is_config_valid = lambda x: True
        # No-op
        self.search_callback = None
        self.synthetic_data_getter = None
        self.past_runs = []
        funcs_processed = set()
        if config_funcs:
            if config_funcs.get("META_search_callback"):
                funcs_processed.add("META_search_callback")
                self.search_callback = config_funcs['META_search_callback']()
                assert isinstance(self.search_callback, SearcherInterface)
                assert inspect.isgeneratorfunction(self.search_callback.generate)
                if config_funcs.get("META_get_synthetic"):
                    funcs_processed.add("META_get_synthetic")
                    self.synthetic_data_getter = config_funcs['META_get_synthetic']
            if config_funcs.get("META_get_no_rebuild_params"):
                funcs_processed.add("META_get_no_rebuild_params")
                norebuild_params = config_funcs.get("META_get_no_rebuild_params")()
                assert isinstance(norebuild_params, list)
                # Make sure these keys all exist in our grid params:
                # But we might not know grid params if a search_callback is being used:
                if self.search_callback is None:
                    missing_keys = set(norebuild_params) - set(griddict.keys())
                    if len(missing_keys) > 0:
                        raise RuntimeError(f"The keys: {missing_keys} were mentioned in META_get_no_rebuild_params, but are not a specified parameter in:\n{griddict.keys()}")
                else:
                    print("WARNING: Not checking get_no_rebuild_params against grid parameters, be careful")
                # For use later, we're gonna turn this into a set:
                self.no_rebuild_params = set(norebuild_params)
            if config_funcs.get("META_is_config_valid"):
                funcs_processed.add("META_is_config_valid")
                self.is_config_valid = config_funcs["META_is_config_valid"]

            # Other META handling goes here
            unmatched_funcs = set(config_funcs.keys()) - funcs_processed
            if len(unmatched_funcs) > 0:
                raise RuntimeError(f"Found the following META functions which haven't been implemented, refer to README for proper naming {unmatched_funcs}")

        # Make sure we can set all keys are in our config:
        base_config_dict = self.base_config.as_dict()
        if not args.no_check_keys:
            for grid_key in griddict.keys():
                if grid_key not in base_config_dict:
                    print(f"{grid_key} not found in base config")
                    print(f"{base_config_dict}")
                    assert False
        # Make sure all values are non-empty lists of something that isn't a list or a dict
        # TODO expand this to something reasonable to help META_search_callback
        if self.search_callback:
            print("WARNING: Skipping parameter validation because META_search_callback was provided")
        else:
            for val in griddict.values():
                assert isinstance(val, list)
                #assert len(val) >= 1
                assert all(not isinstance(el, list) and not isinstance(el, dict) for el in val)
        self.grid = griddict
        if self.search_callback is None:
            self.search_callback = CartesianProduct(self.grid, self.no_rebuild_params)

    def need_to_rebuild(self, old_cross, new_cross):
        if old_cross is None:
            # If we were told that we can use an engine already present, do that
            return not args.use_existing_engine
        if self.no_rebuild_params is None:
            # If we don't know anything else about our problem, we have to rebuild
            return True
        # Sanity check, make sure all keys of one are in the other
        assert set(old_cross.keys()) == set(new_cross.keys())
        # Now we have to see if all of the changed keys are in the "no_rebuild_needed" set:
        changed_keys = {k for k in old_cross.keys() if old_cross[k] != new_cross[k]}
        return len(changed_keys - self.no_rebuild_params) != 0

    def get_stats(self, cross, expect=True):
        """Try and get stats from potentially created artifacts, with the option of masking exceptions
        If required is false and stats cannot be found, None is returned"""
        # Debug path first
        if self.synthetic_data_getter:
            res = self.synthetic_data_getter(cross)
            if res is None:
                raise RuntimeError(f"Expected synthetic results for cross term:\n{cross}")
            return res
        # Common path
        log_dir = self.cross_to_log_dir(cross)
        try:
            stats = PerfStats(directory_or_data=log_dir)
        except Exception as e:
            if expect:
                raise e
            else:
                return False
        return stats

    def cross_to_standalone_config(self, cross_term):
        """ Acquire a standalone config which is usable by generate_engines/run_harness targets."""
        class CrossConfig(self.base_config):
            @classmethod
            def as_dict(cls):
                d = super().as_dict()
                d.update(cross_term)
                return d
        return CrossConfig

    def apply_overriden_configs(self, overrides):
        """ Get the `load_config_fn` that is used by code/main.py to load
            the standalone config with the given overrides applied. """
        def load_config_fn(benchmarks: List[Benchmark], scenarios: List[Scenario]):
            cls = self.cross_to_standalone_config(overrides)
            ConfigRegistry._reset()
            ConfigRegistry.register(*list(self.workload_setting.as_dict().values()))(cls)
        return load_config_fn

    def cross_to_log_dir(self, cross_term):
        """ Find the log directory for a given cross-term.

        Contains basic collision detection. It's up to the user if old data wants to be:
        - Overwritten -> Move/remove problematic directory.
        - Kept -> Increase cross_hash length (will invalidate runs with --use_cached).
        """
        # Very arbitrary naming scheme, but at least reproducible with an identical input grid (if we crash, etc.)
        base_str = f"build/logs/grid_{self.system_id}_{self.benchmark}_{self.scenario}_"
        json_str = json.dumps(cross_term, sort_keys=True).encode("utf-8")
        # Ten chars should be "good_enough".
        cross_hash = hashlib.md5(json_str).hexdigest()[:10]
        path = f"{base_str}{cross_hash}"
        # Let's do a quick collision check:
        metafile_path = os.path.join(path, METAFILE)
        # Try opening the previous run metadata and reading it out
        # If the config/cross_term stored doesn't match our current cross_term, panic
        if os.path.exists(metafile_path):
            with open(metafile_path) as f:
                meta_dict = json.load(f)
            if meta_dict['config'] != cross_term:
                raise RuntimeError(f"Hash collision detected with cross_term: {cross_term}\nTry increasing digest length!")

        return path

    def _should_run(self, cross_term):
        full_config = self.cross_to_standalone_config(cross_term).as_dict()
        try:
            should_run = self.is_config_valid(full_config)
        except Exception as e:
            print("Unknown error: is META_is_config_valid defined correctly")
            raise e
        return should_run

    def __iter__(self):
        """ Iterates through this Config's cross-terms.

        Default behavior will potentially take advantage of no_rebuild_params and is_config_valid for better scheduling (see CartesianProduct)
        """

        if args.dry_run and self.search_callback.is_dynamic:
            print(f"DRY-RUN: Would call into user-provided META_search_callback here.")

        elif self.past_runs:
            for term, stats in self.past_runs:
                yield term
        else:
            generator = self.search_callback.generate()
            past_run = None
            while True:
                if self.past_runs:
                    print("Runs so far:\n======================")
                    for run in self.past_runs:
                        d = run.cross.copy()
                        d.update({"results_validity": run.stats["result_validity"]})
                        pprint(d)
                    print("======================")
                try:
                    term_dict = generator.send(past_run)
                except StopIteration:
                    break
                if self._should_run(term_dict):
                    yield term_dict
                    if not args.dry_run:
                        past_run = Run(term_dict.copy(), self.get_stats(term_dict))
                        self.past_runs.append(past_run)
                elif args.dry_run:
                    print(f"DRY-RUN: Not running {term_dict} because it failed user-provided constraint function")


def py_or_json_opener(thing):
    if thing.endswith('.json'):
        with open(thing, 'r') as f:
            try:
                param_dict = json.load(f)
            except Exception as e:
                print(e)
                raise e
            meta_funcs = None
    elif thing.endswith('.py'):
        try:
            loader = importlib.machinery.SourceFileLoader('mod', thing)
            mod = loader.load_module()
            attrs = [(k, v) for k, v in vars(mod).items() if not k.startswith('_')]
            param_dict = {k: v for k, v in attrs if isinstance(v, list)}
            meta_funcs = {k: v for k, v in attrs if callable(v) and k.startswith("META")}
        except Exception as e:
            print(e)
            raise e
    else:
        raise RuntimeError("Config doesn't end in .py or .json")
    return param_dict, meta_funcs


def tee(cmd):
    # Unused return, but we need to request output to get tee effect
    run_command(cmd, get_output=True, tee=True)


class Temperature:
    """A wrapper, non-instance class to do one-time initialization of get_system().

    Initialization is done at the first call of logged_temp_wait akin to 'magic statics'
    We need run-time initialization (rather than module-load-time) to support spoof_system_id"""
    system = None

    @classmethod
    def _get_core_temps(cls):
        if SystemClassifications.is_xavier(cls.system):
            # Because we don't have nvidia-smi on xavier, we need to use sysfs to read out the temperature
            # The type of the thermal_zone is in /sys/devices/virtual/thermal/termal_zone<N>/type.
            # To avoid doing a bunch of process spawn to check if a given node is a GPU node, we're gonna hardcode the GPU_therm node:
            # AGX_Xavier: thermal_zone1
            # Xavier_NX: thermal_zone1
            # NOTE, this may change in subsequent/previous submission models.
            try:
                out_text = run_command("cat /sys/devices/virtual/thermal/thermal_zone1/temp", get_output=True, tee=False)
                # The temperature is in units of milli degC, so scale the result:
                temps = [int(str_temp) / 1000 for str_temp in out_text]
            except Exception as e:
                print("Bad temp reading")
                raise e
        else:
            # Non-xavier branch
            try:
                out_text = run_command("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", get_output=True, tee=False)
                # multi-gpu instance return a list of strings corresponding to temp of each core
                temps = [int(str_temp) for str_temp in out_text]
            except Exception as e:
                print("Bad temp reading")
                raise e
        return temps

    @classmethod
    def logged_temp_wait(cls, temp, timeout=None):
        """ Spinwait on GPU temperature with optional timeout. Returns last measured temperature

        For multi-GPU systems, we use mean temperature.
        """
        if cls.system is None:
            cls.system = DETECTED_SYSTEM
        start_time = time.perf_counter()
        # Poll with timeout
        succ = False
        while (time.perf_counter() - start_time) < timeout:
            temps = cls._get_core_temps()
            mean_temp = sum(temps) / len(temps)
            if mean_temp <= temp:
                print("GPU has finished cooling")
                succ = True
                break
            print(mean_temp)
            time.sleep(2)
        if not succ:
            print("GPU failed to fully cool")
        return mean_temp


def finalize_log_dir(directory, cross_term, extra_info):
    data = {
        "config": cross_term,
        "run_info": extra_info
    }
    with open(os.path.join(directory, METAFILE), 'w') as f:
        json.dump(data, f, indent=2)


def execute(lam, description):
    if args.dry_run:
        print(f"DRY-RUN: {description}")
    else:
        return lam()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=str, help="Fuzzy MLPerf-I Benchmark name. eg: Resnet50, rnnt, or rnn-t")
    parser.add_argument("scenario", type=str, help="Fuzzy MLPerf-I Scenario name. eg: Offline, offline, Server")
    parser.add_argument("--no_check_keys", action="store_true",
                        help="If set, will not check that a key defined in the config file is present in the config, adding it to the temp per-run config if needed rather than updating.")
    run_group = parser.add_argument_group("running")
    run_group.add_argument("--temp_timeout", default=60, type=int, help="Max number of seconds to sleep between runs (and building) if cool_temp isn't reached")
    run_group.add_argument("--cool_temp", default=50, type=int, help="Target temperature in degC to wait for before building or running")
    run_group.add_argument("--extra_run_args", default='', help="String to insert inside of run_harness and generate_engine's RUN_ARGS")
    run_group.add_argument("--use_cached", action='store_true', help="Don't rerun configurations which have existing artifacts")
    run_group.add_argument("--dry_run", action='store_true', help="Prints actions instead of executing, used for verifying config. Implicitly includes --noparse")
    run_group.add_argument("--use_existing_engine", action='store_true',
                           help="Don't unconditionally build an engine for the first build run. (Note, if there a subsequent run requires building a new engine, a new engine will still be built.)")
    parse_group = parser.add_argument_group("parsing")
    parse_group.add_argument("--noparse", action='store_true', help="Skip the parsing/results stage")
    parse_group.add_argument("--verbose_stats", action='store_true', help="Parse all stats, not just scenario-specific fields")
    # TODO, add additional parsing output formats. For now, csv of all fields is good enough.
    parser.add_argument("config", type=py_or_json_opener, help=".json or .py file specifying parameters and values. See README for schema")
    parser.add_argument("--spoof_system_id", type=str, metavar="SYSTEM_ID", help="Use a given SYSTEM_ID instead of the host's. Can only be used with --dry_run and/or --use_cached")
    args = parser.parse_args()
    if args.spoof_system_id and (not args.dry_run and not args.use_cached):
        parser.error("--spoof_system_id requires --dry_run or --use_cached")
    configpy_vars, configpy_funcs = args.config
    config_grid = ConfigGrid(args.benchmark, args.scenario, configpy_vars, configpy_funcs)
    exec_stats = ExecStats()
    past_cross = None
    for cross_term in config_grid:
        if args.use_cached and config_grid.get_stats(cross_term, expect=False):
            # No-op, just doing this for uniformity
            execute(lambda: None,
                    f"Skip build and run for cached item: {cross_term}")
            exec_stats.cached_runs += 1
            continue
        load_cross_config_fn = config_grid.apply_overriden_configs(cross_term)
        if config_grid.need_to_rebuild(past_cross, cross_term):
            exec_stats.builds += 1
            # If this is our first pass, we don't need to sleep before building an engine
            if past_cross:
                execute(lambda: Temperature.logged_temp_wait(args.cool_temp, args.temp_timeout),
                        f"Wait for temp to cool to {args.cool_temp} (timeout at {args.temp_timeout}sec)")
            main_args = parse_main_args(custom=([
                "--action", "generate_engines",
                "--benchmarks", args.benchmark,
                "--scenarios", args.scenario
            ]))
            # Must set sys.argv for other fields not in `MainArgs`
            sys.argv = sys.argv[:1] + args.extra_run_args.split()
            execute(lambda: main(main_args, config_grid.system, load_config_fn=load_cross_config_fn),
                    f"Build")
        else:
            exec_stats.cached_builds += 1
        # We always sleep before a run:
        start_temp = execute(lambda: Temperature.logged_temp_wait(args.cool_temp, args.temp_timeout),
                             f"Wait for temp to cool to {args.cool_temp} (timeout at {args.temp_timeout}sec)")
        log_dir = config_grid.cross_to_log_dir(cross_term)
        main_args = parse_main_args(custom=([
            "--action", "run_harness",
            "--benchmarks", args.benchmark,
            "--scenarios", args.scenario
        ]))
        # Must set sys.argv for other fields not in `MainArgs`
        sys.argv = sys.argv[:1] + args.extra_run_args.split() + [
            "--test_mode", "PerformanceOnly",
            "--log_dir", log_dir
        ]
        execute(lambda: main(main_args, config_grid.system, load_config_fn=load_cross_config_fn),
                f"Run with cross term {cross_term}")

        # For now, we're not recording too much extra run info, but for now it's done here.
        extra_info = {
            'start_temp': start_temp
        }
        execute(lambda: finalize_log_dir(directory=log_dir, cross_term=cross_term, extra_info=extra_info),
                f"Dump additional run info into log_dir")

        exec_stats.runs += 1
        past_cross = cross_term.copy()
    execute(lambda: None,
            f"Session statistics: {exec_stats}")
    if not args.noparse and not args.dry_run:
        for idx, cross_term in enumerate(config_grid):
            stats = config_grid.get_stats(cross_term).data

            # Throw these stats in the dict to do easy printing. (Use | in python3.9)
            cross_term.update(stats)
            sorted_keys = sorted(cross_term.keys())
            if idx == 0:
                # print schema
                schema_str = ",".join([k for k in sorted_keys])
                print(schema_str)
            out_str = ",".join([str(cross_term[k]) for k in sorted_keys])
            print(out_str)
