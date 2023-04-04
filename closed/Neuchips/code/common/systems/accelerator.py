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

from __future__ import annotations
from abc import ABC, abstractmethod, abstractclassmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Callable, ClassVar, Final, List, Tuple, Union

import re
import shutil
import textwrap
import os

from code.common import run_command
from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.info_source import InfoSource
from code.common.systems.inferentia import INFERENTIA_INFO_SOURCE, Inferentia

SOC_MODEL_FILEPATH: Final[str] = "/sys/firmware/devicetree/base/model"
"""str: Defines the OS file that contains the SoC's name"""

NV_GPU_INFO_COMMAND: Final[str] = "nvidia-smi"
"""str: The executable for desktop/server/non-SoC systems to interact with the GPU. For NVIDIA, this is nvidia-smi"""

NV_GPU_INFO_QUERY_FIELDS: Tuple[str, ...] = (
    "gpu_name",
    "pci.device_id",
    "uuid",
    "memory.total",
    "power.limit",
    "power.max_limit",
    "mig.mode.current",
)

NVIDIA_SMI_GPU_REGEX = re.compile(r"GPU (\d+): ([\w\- ]+) \(UUID: (GPU-[0-9a-f\-]+)\)")
"""
re.Pattern: Regex to match nvidia-smi output for GPU information
            match(1) - GPU index
            match(2) - GPU name
            match(3) - GPU UUID
"""
NVIDIA_SMI_MIG_REGEX = re.compile(
    r"\s+MIG\s+(\d+)g.(\d+)gb\s+Device\s+(\d+):\s+\(UUID:\s+(MIG-[0-9a-f\-]+)\)"
)
"""
re.Pattern: Regex to match nvidia-smi output for MIG information
            match(1) - Number of GPCs in the MIG slice
            match(2) - Allocated video memory capacity in the MIG slice in GB
            match(3) - MIG device ID
            match(4) - MIG instance UUID
"""


def get_accelerator_info(force_no_gpu_cmd: bool = False) -> Dict[Tuple[int, str, str]: List[Tuple[int, str, int, int]]]:
    """Collect info including UUIDs of the GPUs and MIGs if instantiated; since MIGs always instantiated 
       under a specific GPU, MIGs info including UUIDs are populated under its associated GPU's info

    Args:
        force_no_gpu_cmd (bool): Set to True to skip dGPU detection. Only used for testing. This method and arguments
                                 are not exposed outside of the systems module API. (Default: False)

    Returns:
        Dict[Tuple[int, str, str]: List[Tuple[int, str, int, int]]]: Containing various information of GPUs/MIGs
        Ex)
        {
            Tuple (0 == GPU PCI bus index, GPU0's UUID, GPU0's name): 
                [
                    Tuple (0 == GPU0's MIG device ID, MIG0's UUID, MIG0's #GPCs, MIG0's MEM in GB),
                    Tuple (1, MIG1's UUID, MIG1's #GPCs, MIG1's MEM in GB),
                    ...,
                ],
            Tuple (1, GPU1's UUID, GPU1's name): 
                [
                    ...,
                ],
            ...,
        }
    """
    # get list of the populated GPU/MIG instances
    # Format:
    # GPU <gpu id>: <gpu name> (UUID: <GPU UUID>)
    #   MIG <num_gpc>g.<mem_capacity>gb      Device  <i>: (UUID: <MIG UUID>)
    #   ...
    # GPU ...
    #   ...

    # dict to return
    gpu_mig_map = dict()

    if force_no_gpu_cmd or shutil.which(NV_GPU_INFO_COMMAND) is None:
        return gpu_mig_map

    cmd = f"CUDA_VISIBLE_ORDER=PCI_BUS_ID {NV_GPU_INFO_COMMAND} -L"
    info = run_command(cmd, get_output=True, tee=False, verbose=False)

    # MIG instance is always populated under GPU instance, i.e. no MIG can appear without GPU
    current_gpu = None
    for i in info:
        # fill GPU
        g = NVIDIA_SMI_GPU_REGEX.match(i)
        if g:
            gpu_id = int(g.group(1))
            gpu_name = g.group(2)
            gpu_uuid = g.group(3)
            current_gpu = (gpu_id, gpu_uuid, gpu_name)
            assert len(gpu_mig_map) == gpu_id, "GPU found out of order:\nf{info}"
            gpu_mig_map[current_gpu] = list()
        # fill MIG
        else:
            assert current_gpu, "MIG found without its parent GPU:\nf{info}"
            m = NVIDIA_SMI_MIG_REGEX.match(i)
            mig_device_id = int(m.group(3))
            mig_uuid = m.group(4)
            mig_num_gpcs = int(m.group(1))
            mig_mem_gb = int(m.group(2))
            assert len(gpu_mig_map[current_gpu]) == mig_device_id, "MIG found out of order:\nf{info}"
            gpu_mig_map[current_gpu].append((mig_device_id, mig_uuid, mig_num_gpcs, mig_mem_gb))

    return gpu_mig_map


def handle_cuda_visible_devices(instance_list: List[Dict[str, str]],
                                contains_mig: bool = False) -> List[Dict[str, str]]:
    """Select relevant GPU/MIG instance if CUDA_VISIBLE_DEVICES environment variable is set.

    Args:
        instance_list (list): Collection of GPU or MIG instances collected from get_gpu_info/get_mig_info callables
        contains_mig (bool): If False, instance_list contains GPU; If True, instance list contains MIG

    Returns:
        List[Dict[str, str]]: A list where the element at index i represents identifying data about the i-th instance. 
                              An empty list means no GPUs were detected.
    """

    designated_devices_var = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    # nothing to do
    if (not designated_devices_var) or\
            not any(instance_list):
        return instance_list

    # list to return
    filtered_instance_list = list()

    designated_devices = designated_devices_var.split(",")
    encountered_devices = set()
    encountered_gpu_pci_id = set()

    def _process_instance_list_with_uuid(l, key):
        if l.get('uuid') not in encountered_devices:
            filtered_instance_list.append(l)
            encountered_devices.add(l.get('uuid'))
        encountered_gpu_pci_id.add(l.get(key))

    # instance list is likely in good order
    for l in instance_list:
        for i, d in enumerate(designated_devices):
            if d.isnumeric():  # if PCIe BUS index
                if not contains_mig and int(d) == l.get('pci_index'):
                    _process_instance_list_with_uuid(l, 'pci_index')
                elif contains_mig and int(d) == l.get('parent_gpu_index'):
                    _process_instance_list_with_uuid(l, 'parent_gpu_index')
            elif not contains_mig and d.startswith("GPU-"):  # if GPU UUID matches single instance
                if d == l.get('uuid'):
                    _process_instance_list_with_uuid(l, 'pci_index')
            elif contains_mig and d.startswith("MIG-"):  # if MIG UUID matches single instance
                if d == l.get('uuid'):
                    _process_instance_list_with_uuid(l, 'parent_gpu_index')
            elif contains_mig and d.startswith("GPU-"):  # if GPU UUID is given while MIGs are populated
                if d == l.get('parent_gpu_uuid'):
                    _process_instance_list_with_uuid(l, 'parent_gpu_index')
            else:
                raise RuntimeError(f"Unexpected '{d}' from CUDA_VISIBLE_DEVICES: {designated_devices_var}'")

    # due to device ordinal, need to sort upon GPU's PCI index and reassign the index
    # although above instance_list came in good order, and we may not need this sorting, sort to be sure
    filtered_instance_list = sorted(filtered_instance_list,
                                    key=lambda i: (i.get('pci_index'), i.get('parent_gpu_index')))
    for i, j in enumerate(sorted(encountered_gpu_pci_id)):
        for l in filtered_instance_list:
            if l.get('pci_index') == j:
                l['pci_index'] = i
            elif l.get('parent_gpu_index') == j:
                l['parent_gpu_index'] = i
    return filtered_instance_list


def get_compute_sm(device_index: int) -> int:
    if importlib.util.find_spec("pycuda") is None and os.environ.get("OUTSIDE_MLPINF_ENV", "0") == "1":
        return None

    # Force PyCuda to be in PCI Bus order
    old_cuda_order = os.environ.get("CUDA_VISIBLE_ORDER", None)
    os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BUS_ID"

    import pycuda.driver as cuda
    import pycuda.autoinit
    d = cuda.Device(device_index)
    c_major = d.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR)
    c_minor = d.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MINOR)

    if old_cuda_order is None:
        del os.environ["CUDA_VISIBLE_ORDER"]
    else:
        os.environ["CUDA_VISIBLE_ORDER"] = old_cuda_order

    return c_major * 10 + c_minor


def get_gpu_info(soc_model_filepath: str = SOC_MODEL_FILEPATH,
                 force_no_gpu_cmd: bool = False,
                 mig_enabled: bool = False,
                 skip_sm_check: bool = False,
                 skip_cuda_devices_filter: bool = False) -> List[Dict[str, str]]:
    """Attempts to run various commands to retrieve info about the GPUs in the system.

    Args:
        soc_model_filepath (str): The path to the 'model' file containing the name of the SoC's model. This method and
                                  arguments are not exposed outside of the systems module API, as it is only called
                                  internally in GPU_INFO_SOURCE. (Default: SOC_MODEL_FILEPATH)
        force_no_gpu_cmd (bool): Set to True to skip dGPU detection. Only used for testing. This method and arguments
                                 are not exposed outside of the systems module API. (Default: False)
        mig_enabled (bool): Set to true to allow grabbing information for GPUs that have MIG enabled. (Default: False)
        skip_sm_check (bool): Set to True to skip SM detection, since this is detected via PyCUDA and cannot be spoofed.
                              Only used for testing. This method and arguments are not exposed outside of the systems
                              module API.  (Default: False)
        skip_cuda_devices_filter (bool): Set to True to skip filtering GPU instances as per CUDA_VISIBLE_DEVICES environment
                                         variable (Default: False)                              

    Returns:
        List[Dict[str, str]]: A list where the element at index i represents identifying data about the i-th GPU. An
                              empty list means no GPUs were detected.
    """

    # list to return
    info = []

    if not force_no_gpu_cmd and shutil.which(NV_GPU_INFO_COMMAND) is not None:
        query_flag = "--query-gpu=" + ",".join(NV_GPU_INFO_QUERY_FIELDS)
        cmd = f"CUDA_VISIBLE_ORDER=PCI_BUS_ID {NV_GPU_INFO_COMMAND} {query_flag} --format=csv,noheader"
        gpu_info = run_command(cmd, get_output=True, tee=False, verbose=False)

        # Strip empty lines and tokens
        tmp = [[x.strip() for x in line.split(',')] for line in gpu_info if len(line) > 0]
        uuid2index = {data[2]: i for i, data in enumerate(tmp)}

        gpu_info_buffer = []
        found_mig_enabled_gpu = False
        for data in tmp:
            if len(data) != len(NV_GPU_INFO_QUERY_FIELDS):
                raise RuntimeError(f"Internal: Malformed {NV_GPU_INFO_COMMAND} output")
            data_dict = {NV_GPU_INFO_QUERY_FIELDS[i]: data[i].strip() for i in range(len(data))}
            data_dict["pci_index"] = uuid2index[data[2]]
            if data_dict["mig.mode.current"] == "Enabled":
                found_mig_enabled_gpu = True
            gpu_info_buffer.append(data_dict)
        # if MIGs are enabled on some GPUs, only keep those,
        # as we don't have a use case where GPUs and MIGs are used together
        if found_mig_enabled_gpu:
            if mig_enabled:
                gpu_info_buffer = [b for b in gpu_info_buffer if b["mig.mode.current"] == "Enabled"]
            else:
                gpu_info_buffer = []
        info = gpu_info_buffer if skip_cuda_devices_filter else handle_cuda_visible_devices(gpu_info_buffer, False)

        for i in info:
            # WAR for MLPINF-1495:
            # In MIG Mode, PyCUDA can only see GPU0 for some reason. It is likely a bug with PyCUDA and MIG.
            # As a workaround, since this is only for MIG, if mig is enabled, always use 0 as the PCI Index.
            pci_index = 0 if (mig_enabled or i["mig.mode.current"] == "Enabled") else i["pci_index"]
            i["compute_sm"] = None if skip_sm_check else get_compute_sm(pci_index)

    elif os.path.exists(soc_model_filepath):
        with open(soc_model_filepath) as f:
            name = ''.join(c for c in f.read().strip() if c.isprintable())  # Removes the x00 from the end.
        # Jetson SoCs are single-iGPU with 2x DLA. However, we do not detect the DLAs currently.
        igpu = {
            "gpu_name": name,
            "pci.device_id": None,
            "uuid": None,
            "memory.total": None,  # memory total of 'None' represents an iGPU with shared memory
            "power.limit": None,
            "power.max_limit": None,
            "pci_index": 0,
            "compute_sm": None if skip_sm_check else get_compute_sm(0),
        }
        info = [igpu] if skip_cuda_devices_filter else handle_cuda_visible_devices([igpu], False)

    return info


@dataclass(eq=True, frozen=True)
class Accelerator(Hardware):
    name: Union[Matchable, str, AliasedName]
    accelerator_type: Union[Matchable, AcceleratorType]


@dataclass(eq=True, frozen=True)
class GPU(Accelerator):
    vram: Union[Matchable, Memory]
    max_power_limit: Union[Matchable, float]
    pci_id: Union[Matchable, AliasedName]
    compute_sm: Union[Matchable, int]

    @classmethod
    def detect(cls):
        """Grabs the next GPU info from GPU_INFO_SOURCE. The caller must maintain GPU_INFO_SOURCE and make sure it is
        reset before calling it.

        Returns:
            GPU: A GPU object with fields from the top item from the InfoSource buffer
        """
        gpu_fields = next(GPU_INFO_SOURCE)
        name = gpu_fields["gpu_name"]
        total_mem = gpu_fields["memory.total"]
        accelerator_type = AcceleratorType.Integrated if total_mem is None else AcceleratorType.Discrete
        pci_id = gpu_fields["pci.device_id"]
        max_power_limit = gpu_fields["power.max_limit"]
        compute_sm = gpu_fields["compute_sm"]
        vram = None
        if total_mem not in (None, "", "[N/A]", "[Insufficient Permissions]"):
            toks = total_mem.split()
            assert len(toks) <= 2
            if len(toks) == 1:
                # Use 1024-base (KiB, MiB, etc) by convention to copy nvidia-smi
                vram = Memory.to_1024_base(float(toks[0]))
            else:
                suffix = ByteSuffix[toks[1]]
                vram = Memory.to_1024_base(Memory(float(toks[0]), suffix).to_bytes())

        if max_power_limit is not None:
            if max_power_limit == "[N/A]":
                max_power_limit = None
            else:
                max_power_limit = float(max_power_limit.split()[0])
        return GPU(name, accelerator_type, vram, max_power_limit, pci_id, compute_sm)

    def identifiers(self):
        return (
            self.name,
            self.accelerator_type,
            self.vram,
            self.max_power_limit,
            self.pci_id,
            self.compute_sm
        )

    def matches(self, other) -> bool:
        """Matches this GPU with 'other'.

        If other is the same class as self, compare the identifiers. If both have PCI ID set, name is ignored for
        comparison.
        Returns False otherwise.

        Returns:
            bool: Equality based on the rules described above
        """
        if other.__class__ == self.__class__:
            self_id = self.identifiers()
            other_id = other.identifiers()

            # The GPU name can change with driver version. Since name is derived from PCI ID for discrete cards, do not
            # check name if PCI ID is set, since name is unreliable.
            # If PCI IDs are both set, skip name check
            if all(map(lambda x: type(x) in (str, AliasedName), (self_id[4], other_id[4]))):
                return self_id[1:] == other_id[1:]
            else:
                return self_id == other_id
        return NotImplemented

    def __eq__(self, o) -> bool:
        # __eq__ needs to be explicitly defined to override the dataclass __eq__. dataclass.eq needs to be True to
        # auto-generate __hash__.
        return self.matches(o)

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying the data in the GPU

        Returns:
            str: 'Pretty-print' string representation of the GPU
        """
        lines = [f"GPU ({self.pci_id}): {self.name}",
                 f"AcceleratorType: {self.accelerator_type.name}",
                 f"SM Compute Capability: {self.compute_sm}",
                 f"Memory Capacity: {self.vram.pretty_string()}",
                 f"Max Power Limit: {self.max_power_limit} W"]
        s = lines[0] + "\n" + textwrap.indent("\n".join(lines[1:]), " " * 4)
        return s


def get_mig_info(force_no_gpu_cmd=False, skip_sm_check=False) -> List[Dict[str, str]]:
    """Attempts to run various commands to retrieve info about the GPUs in the system.

    Args:
        force_no_gpu_cmd (bool): Set to True to skip dGPU detection. Only used for testing. This method and arguments
                                 are not exposed outside of the systems module API. (Default: False)
        skip_sm_check (bool): Set to True to skip SM detection, since this is detected via PyCUDA and cannot be spoofed.
                              Only used for testing. This method and arguments are not exposed outside of the systems
                              module API.  (Default: False)

    Returns:
        List[Dict[str, str]]: A list where the element at index i represents identifying data about the i-th GPU. An
                              empty list means no GPUs were detected.
    """
    # list to return
    info = []

    if force_no_gpu_cmd or shutil.which(NV_GPU_INFO_COMMAND) is None:
        return info

    # We need to grab information from the parent GPU to get PCI ID
    parent_gpu_info = get_gpu_info(mig_enabled=True,
                                   skip_sm_check=skip_sm_check,
                                   skip_cuda_devices_filter=True)

    extra_info_map = dict()
    for gpu in parent_gpu_info:
        max_power_limit = gpu["power.max_limit"]
        if max_power_limit is not None:
            if max_power_limit == "[N/A]":
                max_power_limit = None
            else:
                max_power_limit = float(max_power_limit.split()[0])
        extra_info_map[gpu["uuid"]] = {
            "pci_id": gpu["pci.device_id"],
            "max_power_limit": max_power_limit,
            "compute_sm": gpu["compute_sm"],
        }

    ACCELERATOR_INFO_SOURCE.reset()
    for gpu, migs in ACCELERATOR_INFO_SOURCE:
        gpu_id, gpu_uuid, gpu_name = gpu
        for mig in migs:
            device_id, uuid, num_gpcs, mem_gb = mig
            mig_slice = {
                "device_id": device_id,
                "uuid": uuid,
                "num_gpcs": num_gpcs,
                "mem_gb": mem_gb,
                "parent_gpu_index": gpu_id,
                "parent_gpu_name": gpu_name,
                "parent_gpu_uuid": gpu_uuid,
            }
            extra_info = extra_info_map[gpu_uuid]
            for k in extra_info:
                mig_slice[f"parent_gpu_{k}"] = extra_info[k]
            info.append(mig_slice)

    return handle_cuda_visible_devices(info, True)


@dataclass(eq=True, frozen=True)
class MIG(GPU):
    num_gpcs: Union[Matchable, int]

    @classmethod
    def detect(cls):
        """Grabs the next MIG instance info from MIG_INFO_SOURCE. The caller must maintain MIG_INFO_SOURCE and make sure
        it is reset before calling it.

        Returns:
            MIG: A MIG object with fields from the top item from the InfoSource buffer
        """
        mig_fields = next(MIG_INFO_SOURCE)
        num_gpcs = mig_fields["num_gpcs"]
        mem_gb = mig_fields["mem_gb"]
        base_gpu_name = mig_fields["parent_gpu_name"]
        name = f"{base_gpu_name} MIG-{num_gpcs}g.{mem_gb}gb"
        vram = Memory(mem_gb, ByteSuffix.GiB)
        accelerator_type = AcceleratorType.Discrete
        pci_id = mig_fields["parent_gpu_pci_id"]
        max_power_limit = mig_fields["parent_gpu_max_power_limit"]
        compute_sm = mig_fields["parent_gpu_compute_sm"]
        return MIG(name, accelerator_type, vram, max_power_limit, pci_id, compute_sm, num_gpcs)

    def identifiers(self):
        return (
            self.name,
            self.accelerator_type,
            self.vram,
            self.max_power_limit,
            self.pci_id,
            self.compute_sm,
            self.num_gpcs,
        )

    def __eq__(self, o) -> bool:
        # __eq__ needs to be explicitly defined to override the dataclass __eq__. dataclass.eq needs to be True to
        # auto-generate __hash__.
        return self.matches(o)

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying the data in the GPU

        Returns:
            str: 'Pretty-print' string representation of the GPU
        """
        lines = [f"{self.num_gpcs}-GPC MIG ({self.pci_id}): {self.name}",
                 f"AcceleratorType: {self.accelerator_type.name}",
                 f"SM Compute Capability: {self.compute_sm}",
                 f"Memory Capacity: {self.vram.pretty_string()}",
                 f"Max Power Limit: {self.max_power_limit} W"]
        s = lines[0] + "\n" + textwrap.indent("\n".join(lines[1:]), " " * 4)
        return s


@dataclass(eq=True, frozen=True)
class AcceleratorConfiguration(Hardware):
    layout: Dict[Accelerator, int]

    def num_gpus(self):
        i = 0
        for accelerator in self.layout:
            if accelerator.__class__ is GPU:
                i += self.layout[accelerator]
        return i

    def num_migs(self):
        i = 0
        for accelerator in self.layout:
            if accelerator.__class__ is MIG:
                i += self.layout[accelerator]
        return i

    def num_inferentia(self):
        i = 0
        for accelerator in self.layout:
            if accelerator.__class__ is Inferentia:
                i += self.layout[accelerator]
        return i

    def get_accelerators(self):
        return list(self.layout.keys())

    def get_primary_accelerator(self):
        if len(self.layout) == 0:
            return None
        return self.get_accelerators()[0]

    @classmethod
    def detect(cls) -> AcceleratorConfiguration:
        """Detects possible known accelerator types and builds a map of accelerator -> count. Known accelerator types
        must be implemented subclasses of 'Hardware'.

        Returns:
            AcceleratorConfiguration: An AcceleratorConfiguration from runtime data
        """
        layout = defaultdict(int)

        # Detect GPUs first
        GPU_INFO_SOURCE.reset()
        while GPU_INFO_SOURCE.has_next():
            gpu = GPU.detect()
            layout[gpu] += 1

        # Detect MIGs
        MIG_INFO_SOURCE.reset()
        while MIG_INFO_SOURCE.has_next():
            mig = MIG.detect()
            layout[mig] += 1

        # Detect Inferentias
        INFERENTIA_INFO_SOURCE.reset()
        while INFERENTIA_INFO_SOURCE.has_next():
            inferentia = Inferentia.detect()
            layout[inferentia] += 1

        return AcceleratorConfiguration(layout)

    def matches(self, other) -> bool:
        if other.__class__ == self.__class__:
            if len(self.layout) != len(other.layout):
                return False

            for accelerator, count in self.layout.items():
                # We actually have to iterate through each accelerator in other in case of Matchables. hashes are not guaranteed
                # to be equivalent even if a.matches(b).
                found = False
                for other_accelerator, other_count in other.layout.items():
                    if accelerator == other_accelerator:
                        found = True
                        if count != other_count:
                            return False
                        break
                if not found:
                    return False
            return True
        return NotImplemented

    def __eq__(self, o) -> bool:
        return self.matches(o)

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying the data in the AcceleratorConfiguration.

        Returns:
            str: 'Pretty-print' string representation of the AcceleratorConfiguration
        """
        lines = ["AcceleratorConfiguration:"]
        for accelerator, count in self.layout.items():
            lines.append(f"{count}x " + accelerator.pretty_string())
        s = lines[0] + "\n" + textwrap.indent("\n".join(lines[1:]), " " * 4)
        return s


ACCELERATOR_INFO_SOURCE: Final[InfoSource] = InfoSource(get_accelerator_info)
GPU_INFO_SOURCE: Final[InfoSource] = InfoSource(get_gpu_info)
MIG_INFO_SOURCE: Final[InfoSource] = InfoSource(get_mig_info)
