#! /usr/bin/env python3
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

import os
import sys
import argparse
import json

import collections

trt_version = "TensorRT 8.5.3"
cuda_version = "CUDA 11.8"
cudnn_version = "cuDNN 8.7.0.84"
dali_version = "DALI 0.31.0"
triton_version = "Triton 22.04"
os_version = "Ubuntu 20.04.4"
driver_version = "Driver 525.60.13"
submitter = "NVIDIA"

soc_sw_version_dict = \
    {
        "orin-jetson": {
            "trt": "TensorRT 8.5.0",
            "cuda": "CUDA 11.4",
            "cudnn": "cuDNN 8.5.0",
            "jetpack": "22.08 Jetson CUDA-X AI Developer Preview",
            "os": "Ubuntu 20.04",
        }
    }


def get_soc_sw_version(accelerator_name, software_name):
    if software_name not in ["trt", "cuda", "cudnn", "jetpack", "os"]:
        raise KeyError(f"No version info for {software_name}. Options: {list(list(soc_sw_version_dict)[0].keys())}")
    if "orin" in accelerator_name.lower():
        # For v2.0 submission, "orin" stands for "orin-jetson"
        if "auto" not in accelerator_name.lower():
            return soc_sw_version_dict["orin-jetson"][software_name]
        else:
            raise KeyError("Only Jetson is available in the Orin family now.")
    else:
        raise KeyError(f"No version info for {accelerator_name}.")


class Status:
    AVAILABLE = "available"
    PREVIEW = "preview"
    RDI = "rdi"


class Division:
    CLOSED = "closed"
    OPEN = "open"
    NETWORK = "network"


class SystemType:
    EDGE = "edge"
    DATACENTER = "datacenter"
    BOTH = "datacenter,edge"

# List of Machines


Machine = collections.namedtuple("Machine", [
    "status",
    "host_processor_model_name",
    "host_processors_per_node",
    "host_processor_core_count",
    "host_memory_capacity",
    "host_storage_capacity",
    "host_storage_type",
    "host_networking",
    "host_networking_topology",
    "accelerator_model_name",
    "accelerator_short_name",
    "mig_short_name",
    "accelerator_memory_capacity",
    "accelerator_memory_configuration",
    "hw_notes",
    "sw_notes",
    "system_id_prefix",
    "system_name_prefix",
    "number_of_type_nics_installed",
    "nics_enabled_firmware",
    "nics_enabled_os",
    "nics_enabled_connected",
    "network_speed_mbit",
    "is_network",
    "network_type",
    "network_media",
    "network_rate",
    "nic_loadgen",
    "number_nic_loadgen",
    "net_software_stack_loadgen",
    "network_protocol",
    "number_connections",
    "nic_sut",
    "number_nic_sut",
    "net_software_stack_sut",
    "network_topology",
    "nic_id",
    "nic_count",
])

# The DGX-A100-640G used as SUT node
LUNA_PROD_72_80GB = Machine(
    status=Status.AVAILABLE,
    host_processor_model_name="AMD EPYC 7742",
    host_processors_per_node=2,
    host_processor_core_count=64,
    host_memory_capacity="2 TB",
    host_storage_capacity="15 TB",
    host_storage_type="NVMe SSD",
    host_networking="",
    host_networking_topology="",
    accelerator_model_name="NVIDIA A100-SXM-80GB",
    accelerator_short_name="A100-SXM-80GB",
    mig_short_name="",
    accelerator_memory_capacity="80 GB",
    accelerator_memory_configuration="HBM2e",
    hw_notes="SUT node with A100 SXM x8 and CX6 x8",
    sw_notes="MOFED 5.7-1.0.2.0",
    system_id_prefix="SUT_DGX",
    system_name_prefix="NVIDIA DGX A100",
    number_of_type_nics_installed="",
    nics_enabled_firmware="",
    nics_enabled_os="",
    nics_enabled_connected="",
    network_speed_mbit="",
    is_network="True",
    network_type="InfiniBand for data exchange, Ethernet for Synchronization between LON and SUT nodes",
    network_media="Optical",
    network_rate="200Gbits/sec",
    nic_loadgen="NVIDIA (Mellanox) CX-6, I350 Gigabit Ethernet NIC for synchronization",
    number_nic_loadgen="8 CX-6 InfiniBand NICs for data exchange, 1 Ethernet NIC for synchronization",
    net_software_stack_loadgen="Mellanox OFED (MOFED) 5.7-1.0.2.0 for InfiniBand, Linux kernel TCP stack 5.4.0-84",
    network_protocol="InfiniBand verbs for data exchange, TCP/IPv4 over Ethernet for synchronization",
    number_connections="8 for data exchange through InfiniBand, 1 for synchronization through Ethernet",
    nic_sut="NVIDIA (Mellanox) CX-6 for data exchange, NVIDIA (Mellanox) CX-4 for synchronization",
    number_nic_sut="8 CX-6 InfiniBand NICs for data exchange, 1 Ethernet NIC for synchronization",
    net_software_stack_sut="Mellanox OFED (MOFED) 5.7-1.0.2.0 for InfiniBand, Linux kernel TCP stack 5.4.0-84",
    network_topology="LON node connected to SUT node through Mellanox Infiniband Switch MQM8700 for data exchange, "
                     "LON node connected to SUT node through Ethernet Switch for synchronization",
    nic_id="CX6",
    nic_count="8",
)


class System():
    def __init__(self, machine, division, system_type, gpu_count=1, mig_count=0, is_triton=False, is_soc=False, is_maxq=False, additional_config=""):
        self.attr = {
            "system_id": self._get_system_id(machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config),
            "submitter": submitter,
            "division": division,
            "system_type": system_type,
            "status": machine.status if division in [Division.CLOSED, Division.NETWORK] else Status.RDI,
            "system_name": self._get_system_name(machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config),
            "number_of_nodes": 1,
            "host_processor_model_name": machine.host_processor_model_name,
            "host_processors_per_node": machine.host_processors_per_node,
            "host_processor_core_count": machine.host_processor_core_count,
            "host_processor_frequency": "",
            "host_processor_caches": "",
            "host_processor_interconnect": "",
            "host_memory_configuration": "",
            "host_memory_capacity": machine.host_memory_capacity,
            "host_storage_capacity": machine.host_storage_capacity,
            "host_storage_type": machine.host_storage_type,
            "host_networking": machine.host_networking,
            "host_networking_topology": machine.host_networking_topology,
            "accelerators_per_node": gpu_count,
            "accelerator_model_name": self._get_accelerator_model_name(machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config),
            "accelerator_frequency": "",
            "accelerator_host_interconnect": "",
            "accelerator_interconnect": "",
            "accelerator_interconnect_topology": "",
            "accelerator_memory_capacity": machine.accelerator_memory_capacity,
            "accelerator_memory_configuration": machine.accelerator_memory_configuration,
            "accelerator_on-chip_memories": "",
            "cooling": "",
            "hw_notes": machine.hw_notes,
            "sw_notes": machine.sw_notes,
            "framework": self._get_framework(machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config),
            "operating_system": os_version if not is_soc else get_soc_sw_version(machine.accelerator_model_name, "os"),
            "other_software_stack": self._get_software_stack(machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config),
            "power_management": "",
            "filesystem": "",
            "boot_firmware_version": "",
            "management_firmware_version": "",
            "other_hardware": "",
            "number_of_type_nics_installed": machine.number_of_type_nics_installed,
            "nics_enabled_firmware": machine.nics_enabled_firmware,
            "nics_enabled_os": machine.nics_enabled_os,
            "nics_enabled_connected": machine.nics_enabled_connected,
            "network_speed_mbit": machine.network_speed_mbit,
            "power_supply_quantity_and_rating_watts": "",
            "power_supply_details": "",
            "disk_drives": "",
            "disk_controllers": "",
            "is_network": machine.is_network,
            "network_type": machine.network_type,
            "network_media": machine.network_media,
            "network_rate": machine.network_rate,
            "nic_loadgen": machine.nic_loadgen,
            "number_nic_loadgen": machine.number_nic_loadgen,
            "net_software_stack_loadgen": machine.net_software_stack_loadgen,
            "network_protocol": machine.network_protocol,
            "number_connections": machine.number_connections,
            "nic_sut": machine.nic_sut,
            "number_nic_sut": machine.number_nic_sut,
            "net_software_stack_sut": machine.net_software_stack_sut,
            "network_topology": machine.network_topology,
        }

    def _get_system_id(self, machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config):
        return "".join([
            (machine.system_id_prefix + "_") if machine.system_id_prefix != "" else "",
            machine.accelerator_short_name,
            ("x" + str(gpu_count)) if not is_soc and mig_count == 0 else "",
            "-MIG_{:}x{:}".format(mig_count * gpu_count, machine.mig_short_name) if mig_count > 0 else "",
            "_{:}x{:}".format(machine.nic_id, machine.nic_count),
            "_TRT" if division in [Division.CLOSED, Division.NETWORK] else "",
            "_Triton" if is_triton else "",
            "_MaxQ" if is_maxq else "",
            "_{:}".format(additional_config) if additional_config != "" else "",
        ])

    def _get_system_name(self, machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config):
        system_details = []
        if not is_soc:
            system_details.append("{:d}x {:}{:}".format(
                gpu_count,
                machine.accelerator_short_name,
                "-MIG-{:}x{:}".format(mig_count, machine.mig_short_name) if mig_count > 0 else ""
            ))
        if is_maxq:
            system_details.append("MaxQ")
        if division in [Division.CLOSED, Division.NETWORK]:
            system_details.append("TensorRT")
        if is_triton:
            system_details.append("Triton")
        if additional_config != "":
            system_details.append(additional_config)
        return "{:} ({:})".format(
            machine.system_name_prefix,
            ", ".join(system_details)
        )

    def _get_accelerator_model_name(self, machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config):
        return "{:}{:}".format(
            machine.accelerator_model_name,
            " ({:d}x{:} MIG)".format(mig_count, machine.mig_short_name) if mig_count > 0 else "",
        )

    def _get_framework(self, machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config):
        frameworks = []
        if is_soc:
            frameworks.append(get_soc_sw_version(machine.accelerator_model_name, "jetpack"))
        if division in [Division.CLOSED, Division.NETWORK]:
            # Distinguish different TRT version based on the arch/model
            if is_soc:
                version = get_soc_sw_version(machine.accelerator_model_name, "trt")
            else:
                version = trt_version
            frameworks.append(version)
        frameworks.append(cuda_version if not is_soc else get_soc_sw_version(machine.accelerator_model_name, "cuda"))
        return ", ".join(frameworks)

    def _get_software_stack(self, machine, division, gpu_count, mig_count, is_triton, is_soc, is_maxq, additional_config):
        frameworks = []
        if is_soc:
            frameworks.append(get_soc_sw_version(machine.accelerator_model_name, "jetpack"))
        if division in [Division.CLOSED, Division.NETWORK]:
            # Distinguish different TRT version based on the arch/model
            if is_soc:
                version = get_soc_sw_version(machine.accelerator_model_name, "trt")
            else:
                version = trt_version
            frameworks.append(version)
        frameworks.append(cuda_version if not is_soc else get_soc_sw_version(machine.accelerator_model_name, "cuda"))
        if division in [Division.CLOSED, Division.NETWORK]:
            frameworks.append(cudnn_version if not is_soc else get_soc_sw_version(machine.accelerator_model_name, "cudnn"))
        if not is_soc:
            frameworks.append(driver_version)
        if division in [Division.CLOSED, Division.NETWORK]:
            frameworks.append(dali_version)
        if is_triton:
            frameworks.append(triton_version)
        return ", ".join(frameworks)

    def __getitem__(self, key):
        return self.attr[key]


submission_systems = [
    # Network submissions
    System(LUNA_PROD_72_80GB, Division.NETWORK, SystemType.DATACENTER, 8, 0, False, False),
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv", "-o",
        help="Specifies the output tab-separated file for system descriptions.",
        default="systems/system_descriptions.tsv"
    )
    parser.add_argument(
        "--dry_run",
        help="Don't actually copy files, just log the actions taken.",
        action="store_true"
    )
    parser.add_argument(
        "--manual_system_json",
        help="Path to the system json that is manually to the system description table.",
        nargs='+',
        default=[]
    )
    return parser.parse_args()


def main():
    args = get_args()

    tsv_file = args.tsv

    summary = []
    for system in submission_systems:
        json_file = os.path.join("..", "..", system["division"], system["submitter"], "systems", "{:}.json".format(system["system_id"]))
        print("Generating {:}".format(json_file))
        summary.append("\t".join([str(i) for i in [
            system["system_name"],
            system["system_id"],
            system["submitter"],
            system["division"],
            system["system_type"],
            system["status"],
            system["number_of_nodes"],
            system["host_processor_model_name"],
            system["host_processors_per_node"],
            system["host_processor_core_count"],
            system["host_processor_frequency"],
            system["host_processor_caches"],
            system["host_processor_interconnect"],
            system["host_memory_configuration"],
            system["host_memory_capacity"],
            system["host_storage_capacity"],
            system["host_storage_type"],
            system["host_networking"],
            system["host_networking_topology"],
            system["accelerators_per_node"],
            system["accelerator_model_name"],
            system["accelerator_frequency"],
            system["accelerator_host_interconnect"],
            system["accelerator_interconnect"],
            system["accelerator_interconnect_topology"],
            system["accelerator_memory_capacity"],
            system["accelerator_memory_configuration"],
            system["accelerator_on-chip_memories"],
            system["cooling"],
            system["hw_notes"],
            system["framework"],
            system["operating_system"],
            system["other_software_stack"],
            system["sw_notes"],
            system["power_management"],
            system["filesystem"],
            system["boot_firmware_version"],
            system["management_firmware_version"],
            system["other_hardware"],
            system["number_of_type_nics_installed"],
            system["nics_enabled_firmware"],
            system["nics_enabled_os"],
            system["nics_enabled_connected"],
            system["network_speed_mbit"],
            system["power_supply_quantity_and_rating_watts"],
            system["power_supply_details"],
            system["disk_drives"],
            system["disk_controllers"],
            system["is_network"],
            system["network_type"],
            system["network_media"],
            system["network_rate"],
            system["nic_loadgen"],
            system["number_nic_loadgen"],
            system["net_software_stack_loadgen"],
            system["network_protocol"],
            system["number_connections"],
            system["nic_sut"],
            system["number_nic_sut"],
            system["net_software_stack_sut"],
            system["network_topology"],
        ]]))
        del system.attr["system_id"]
        if not args.dry_run:
            with open(json_file, "w") as f:
                json.dump(system.attr, f, indent=4, sort_keys=True)
        else:
            print(json.dumps(system.attr, indent=4, sort_keys=True))

    # Add the systems to the summary, reading from the json file that's manually written.
    # Note: this is added since Triton system cannot be generated using this script.
    for fpath in args.manual_system_json:
        with open(fpath, "r") as fh:
            print(f"Adding {fpath} manually to the system description table.")
            system = json.load(fh)
            # Read the system_id directly from the file name.
            system_id = fpath.split("/")[-1].split(".")[0]
            summary.append("\t".join([str(i) for i in [
                system["system_name"],
                system_id,
                system["submitter"],
                system["division"],
                system["system_type"],
                system["status"],
                system["number_of_nodes"],
                system["host_processor_model_name"],
                system["host_processors_per_node"],
                system["host_processor_core_count"],
                system["host_processor_frequency"],
                system["host_processor_caches"],
                system["host_processor_interconnect"],
                system["host_memory_configuration"],
                system["host_memory_capacity"],
                system["host_storage_capacity"],
                system["host_storage_type"],
                system["host_networking"],
                system["host_networking_topology"],
                system["accelerators_per_node"],
                system["accelerator_model_name"],
                system["accelerator_frequency"],
                system["accelerator_host_interconnect"],
                system["accelerator_interconnect"],
                system["accelerator_interconnect_topology"],
                system["accelerator_memory_capacity"],
                system["accelerator_memory_configuration"],
                system["accelerator_on-chip_memories"],
                system["cooling"],
                system["hw_notes"],
                system["framework"],
                system["operating_system"],
                system["other_software_stack"],
                system["sw_notes"],
                system["power_management"],
                system["filesystem"],
                system["boot_firmware_version"],
                system["management_firmware_version"],
                system["other_hardware"],
                system["number_of_type_nics_installed"],
                system["nics_enabled_firmware"],
                system["nics_enabled_os"],
                system["nics_enabled_connected"],
                system["network_speed_mbit"],
                system["power_supply_quantity_and_rating_watts"],
                system["power_supply_details"],
                system["disk_drives"],
                system["disk_controllers"],
                system["is_network"],
                system["network_type"],
                system["network_media"],
                system["network_rate"],
                system["nic_loadgen"],
                system["number_nic_loadgen"],
                system["net_software_stack_loadgen"],
                system["network_protocol"],
                system["number_connections"],
                system["nic_sut"],
                system["number_nic_sut"],
                system["net_software_stack_sut"],
                system["network_topology"],
            ]]))

    print("Generating system description summary to {:}".format(tsv_file))
    if not args.dry_run:
        with open(tsv_file, "w") as f:
            for item in summary:
                print(item, file=f)
    else:
        print("\n".join(summary))

    print("Done!")


if __name__ == '__main__':
    main()
