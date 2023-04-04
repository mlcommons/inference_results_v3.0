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

__doc__ = """Script used to store, retrieve, and check connectivity to the IP addresses of Power Meter servers."""

import argparse
import os
import subprocess
import sys

# Old power server
POWER_SERVER_IP_MAP_WINDOWS = {
    "altra-g242-p31-01": "10.176.8.113",  # ipp2-2453
    "ipp1-1469": "10.117.21.175",  # ipp1-2564
    "sjc1-luna-02": "10.117.23.135",  # ipp1-2569
    "ipp1-2468-jetson": "10.117.19.74",  # ipp1-2570
    "ipp1-2469-jetson": "10.117.23.201",  # ipp1-0098
    "ipp1-1470": "10.117.20.125",  # ipp1-2562
}

# mlpinf-workstation-2 will automatically be resolved to the static IP for the power server.
# Try 10.110.39.34 if it's not working
POWER_SERVER_IP_MAP = {
    "altra-g242-p31-01": "mlpinf-workstation-2",
    "ipp1-1469": "mlpinf-workstation-2",
    "sjc1-luna-02": "mlpinf-workstation-2",
    "ipp1-2468-jetson": "mlpinf-workstation-2",
    "ipp1-2469-jetson": "mlpinf-workstation-2",
    "ipp1-1470": "mlpinf-workstation-2",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sut_hostname",
        help="Hostname of the SUT that power is measured for"
    )
    parser.add_argument(
        "--check_connectivity",
        help="Check connectivity status and SSH login validity, instead of printing the IP address.",
        action="store_true"
    )
    parser.add_argument(
        "--use_win",
        help="If set, use windows power server.",
        type=int,
        default=0,
    )
    return parser.parse_args()


def check_connectivity(ip_addr):
    ssh_command = f"ping -c 1 {ip_addr}"
    return subprocess.call(ssh_command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, shell=True)


if __name__ == "__main__":
    args = get_args()
    if args.sut_hostname not in POWER_SERVER_IP_MAP:
        sys.exit(1)

    if args.use_win:
        ip_addr = POWER_SERVER_IP_MAP_WINDOWS[args.sut_hostname]
    else:
        ip_addr = POWER_SERVER_IP_MAP[args.sut_hostname]

    if args.check_connectivity:
        sys.exit(check_connectivity(ip_addr))
    else:
        print(ip_addr)
