/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gflags/gflags.h>

// flags used in network division
DEFINE_string(
    lon_numa_config, "", "NUMA settings: each NUMA node contains a tuple of NIC device name, GPU and CPU indices");
DEFINE_string(
    sut_numa_config, "", "NUMA settings: each NUMA node contains a tuple of NIC device name, GPU and CPU indices");
DEFINE_string(nic_mapping, "", "SmartNIC mapping between LON and SUT nodes");
DEFINE_string(sut_nic_gpu_affinity, "", "Affinity between SmartNICs and GPUs, in the SUT node");
DEFINE_string(lon_netid, "", "LON node network ID (hostname or IPv4 address)");
DEFINE_string(sut_netid, "", "SUT node network ID (hostname or IPv4 address)");
DEFINE_string(tcp_port, "4567", "TCP port SUT node listens on");
DEFINE_bool(
    enable_max_transactions_per_connection, false, "Limit number of transactions sent to one NIC to NIC connection");
DEFINE_uint64(max_transactions_per_connection, 1,
    "How many transactions in series sent to one NIC to NIC connection; default == 1");
DEFINE_uint64(num_ibqps_per_nic, 1, "Number of Infiniband Queue Pairs per NIC; default == 1");
DEFINE_uint64(
    max_wait_before_sending_us, 10, "How long LON waits collecting requests before sending them, in microseconds");
DEFINE_bool(lon_uses_one_issue_queue, false, "LON uses single issue queue");
DEFINE_bool(round_robin_samples_to_multi_issue_queue, false,
    "If LON does not use single issue queue, round-robin samples to multiple issue queues");
DEFINE_bool(smart_balance_samples_to_multi_issue_queue, false,
    "If LON does not use single issue queue, balance samples to multiple issue queues by number of transactions on the "
    "fly");
DEFINE_bool(SUT_uses_host_mem_for_RDMA, false, "SUT uses host memory for RDMA transfers");
