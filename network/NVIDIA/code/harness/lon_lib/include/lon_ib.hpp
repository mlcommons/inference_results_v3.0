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

#ifndef __LON_IB_HPP__
#define __LON_IB_HPP__

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <byteswap.h>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <endian.h>
#include <errno.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <netdb.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include <cuda_runtime_api.h>
#include <infiniband/verbs.h>

#include "utils.hpp"

// suppress assertion
#ifndef NDEBUG
#define NDEBUG
#endif

// some parameters
#define DEF_Q_SIZE (32)
#define DEF_BUF_SIZE (64 * 1024)

// MAX transfer bulk size
#define MAX_BULK_SIZE (1024)

// MAX WRs MLX NIC accepts in RC QP
#define MLX_MAX_WR (8192)

// For smaller payload, inline gives better throughput
#define INLINE_SIZE_THRESHOLD (16)

// ERROR
#define IB_ERR(condition, verb, ret, errno)                                                                            \
    {                                                                                                                  \
        CHECK(condition) << " IBV ERROR: " << verb << " returned " << ret << ", err " << strerror(errno);              \
    }

// 8GB memory provisioned for QSL; maybe using 32GB - 34359738368 or 64GB - 68719476736 if anything fails
std::size_t const PROVISIONED_SEND_BUF_SIZE(8589934592);

// Details for the Infiniband low-level APIs - ibverbs can be found from:
// RDMA Aware Networks Programming User Manual
// https://docs.nvidia.com/networking/display/RDMAAwareProgrammingv17/RDMA+Aware+Networks+Programming+User+Manual
// FWIW More user friendly RDMA_CM API is also available
namespace lonib
{
// handling endianess for 64bit datatype
const uint8_t IsLittleEndian = char(0x0001);
static inline uint64_t htonll(uint64_t x)
{
    return IsLittleEndian ? bswap_64(x) : x;
}
static inline uint64_t ntohll(uint64_t x)
{
    return IsLittleEndian ? bswap_64(x) : x;
}

// forward declaration
class IBConfigs;
class IBDevice;
class IBDevices;
class IBResources;
class IBConnection;
class IBConnections;
class IBOffsetManager;

typedef std::shared_ptr<IBDevice> IBDev_ptr_t;
typedef std::shared_ptr<IBResources> IBRes_ptr_t;
typedef std::shared_ptr<IBOffsetManager> IBOM_ptr_t;
typedef std::shared_ptr<IBConnection> IBCon_ptr_t;
typedef std::shared_ptr<IBConnections> IBCs_ptr_t;
typedef std::tuple<std::pair<std::string, int>, std::pair<std::string, int>, bool, bool, bool, bool, int32_t, int32_t,
    int32_t, std::size_t, uint8_t, std::size_t, std::size_t, uint8_t, std::size_t>
    conn_param_t;

// Offset Manager will track unique ID (QuerySampleResponse::ResponseId)
typedef uintptr_t IBOMelem_t;

typedef std::tuple<uint64_t, uint64_t, uint32_t> IBWC_data_t;
typedef std::shared_ptr<std::array<ibv_wc, MAX_BULK_SIZE>> IBWC_arr_ptr_t;

// data to exchange for establishing the QP connection
typedef struct __attribute__((packed, aligned(4))) IB_connection_property_type
{
    uint64_t addr;   // exchange buffer address
    uint32_t rkey;   // remote side key
    uint32_t qp_num; // QP number
    uint16_t lid;    // local identifier of HSA port
    uint16_t canary; // gollygoop
} lon_ib_con_t;

class IBConfigs
{
private:
    std::string device_name;
    bool is_SUT;
    bool SUT_recv_with_host_mem_for_RDMA;
    bool SUT_send_with_host_mem_for_RDMA;
    bool LON_memory_staging;
    bool CQ_wait_event;
    int ib_port;
    int gpu_idx;
    int qp_idx;
    int32_t NUMA_node;
    int32_t num_NUMA_nodes;
    std::size_t send_queue_size;
    std::size_t send_buffer_size;
    uint8_t send_elem_num;
    std::size_t send_elem_size;
    std::size_t recv_queue_size;
    std::size_t recv_buffer_size;
    uint8_t recv_elem_num;
    std::size_t recv_elem_size;

public:
    IBConfigs()
        : device_name("")
        , is_SUT(false)
        , SUT_recv_with_host_mem_for_RDMA(false)
        , SUT_send_with_host_mem_for_RDMA(false)
        , LON_memory_staging(false)
        , CQ_wait_event(true)
        , ib_port(1)
        , gpu_idx(0)
        , qp_idx(0)
        , NUMA_node(0)
        , num_NUMA_nodes(1)
        , send_queue_size(DEF_Q_SIZE)
        , send_elem_num(1)
        , send_elem_size(DEF_BUF_SIZE / DEF_Q_SIZE)
        , send_buffer_size(DEF_BUF_SIZE)
        , recv_queue_size(DEF_Q_SIZE)
        , recv_elem_num(1)
        , recv_elem_size(DEF_BUF_SIZE / DEF_Q_SIZE)
        , recv_buffer_size(DEF_BUF_SIZE)
    {
    }

    IBConfigs(std::string& device_name_arg, bool is_SUT_arg, bool SUT_recv_with_host_mem_for_RDMA_arg,
        bool SUT_send_with_host_mem_for_RDMA_arg, bool LON_memory_staging_arg, bool CQ_wait_event_arg, int ib_port_arg,
        int gpu_idx_arg, int qp_idx_arg, int32_t NUMA_node_arg, int32_t num_NUMA_nodes_arg,
        std::size_t send_queue_size_arg, uint8_t send_elem_num_arg, std::size_t send_elem_size_arg,
        std::size_t send_buffer_size_arg, std::size_t recv_queue_size_arg, uint8_t recv_elem_num_arg,
        std::size_t recv_elem_size_arg, std::size_t recv_buffer_size_arg)
        : device_name(device_name_arg)
        , is_SUT(is_SUT_arg)
        , SUT_recv_with_host_mem_for_RDMA(SUT_recv_with_host_mem_for_RDMA_arg)
        , SUT_send_with_host_mem_for_RDMA(SUT_send_with_host_mem_for_RDMA_arg)
        , LON_memory_staging(LON_memory_staging_arg)
        , CQ_wait_event(CQ_wait_event_arg)
        , ib_port(ib_port_arg)
        , gpu_idx(gpu_idx_arg)
        , qp_idx(qp_idx_arg)
        , NUMA_node(NUMA_node_arg)
        , num_NUMA_nodes(num_NUMA_nodes_arg)
        , send_queue_size(send_queue_size_arg)
        , send_elem_num(send_elem_num_arg)
        , send_elem_size(send_elem_size_arg)
        , send_buffer_size(send_buffer_size_arg)
        , recv_queue_size(recv_queue_size_arg)
        , recv_elem_num(recv_elem_num_arg)
        , recv_elem_size(recv_elem_size_arg)
        , recv_buffer_size(recv_buffer_size_arg)
    {
    }

    ~IBConfigs() {}

    void print_configs()
    {
        std::stringstream stream;
        stream << std::endl
               << (is_SUT ? "SUT" : "LON") << std::endl
               << "  Device name     : " << device_name << std::endl
               << "  IB port         : " << ib_port << std::endl
               << "  assoc'd GPU idx : " << gpu_idx << std::endl
               << "  QP idx          : " << qp_idx << std::endl
               << "  RDMA rcv hostmem: " << std::boolalpha << SUT_recv_with_host_mem_for_RDMA << std::endl
               << "  RDMA snd hostmem: " << std::boolalpha << SUT_send_with_host_mem_for_RDMA << std::endl
               << "  LON mem staging : " << std::boolalpha << LON_memory_staging << std::endl
               << "  CQ wait event   : " << std::boolalpha << CQ_wait_event << std::endl
               << "  Send queue size : " << send_queue_size << std::endl
               << "  Send buffer size: " << send_buffer_size << std::endl
               << "  Recv queue size : " << recv_queue_size << std::endl
               << "  Recv buffer size: " << recv_buffer_size << std::endl
               << " ------------------------------------------------" << std::endl;
        gLogVerbose << stream.str();
    }

    const std::string& get_device_name() const
    {
        return device_name;
    }
    bool get_is_SUT()
    {
        return is_SUT;
    }
    bool get_SUT_recv_with_host_mem_for_RDMA()
    {
        return SUT_recv_with_host_mem_for_RDMA;
    }
    bool get_SUT_send_with_host_mem_for_RDMA()
    {
        return SUT_send_with_host_mem_for_RDMA;
    }
    bool get_LON_memory_staging()
    {
        return LON_memory_staging;
    }
    bool get_CQ_wait_event()
    {
        return CQ_wait_event;
    }
    int get_ib_port()
    {
        return ib_port;
    }
    int get_gpu_idx()
    {
        return gpu_idx;
    }
    int get_qp_idx()
    {
        return qp_idx;
    }
    int32_t get_NUMA_node()
    {
        return NUMA_node;
    }
    int32_t get_num_NUMA_nodes()
    {
        return num_NUMA_nodes;
    }
    std::size_t get_send_queue_size()
    {
        return send_queue_size;
    }
    std::size_t get_recv_queue_size()
    {
        return recv_queue_size;
    }
    std::size_t get_send_buffer_size()
    {
        return send_buffer_size;
    }
    uint8_t get_send_elem_num()
    {
        return send_elem_num;
    }
    std::size_t get_send_elem_size()
    {
        return send_elem_size;
    }
    std::size_t get_recv_buffer_size()
    {
        return recv_buffer_size;
    }
    uint8_t get_recv_elem_num()
    {
        return recv_elem_num;
    }
    std::size_t get_recv_elem_size()
    {
        return recv_elem_size;
    }
};

class IBDevice
{
private:
    std::string device_name;
    ibv_device* device;
    ibv_context* device_ctx;
    ibv_device_attr device_attr;

public:
    IBDevice(ibv_device* dev_arg)
        : device(dev_arg)
        , device_name(std::string(ibv_get_device_name(dev_arg)))
        , device_ctx(ibv_open_device(dev_arg))
    {
        // query device attributes/capabilities and store into device_attr
        int ret{0};
        ret = ibv_query_device(device_ctx, &device_attr);
        IB_ERR(!ret, "ibv_query_device", ret, errno);
    }

    ~IBDevice()
    {
        ibv_close_device(device_ctx);
    }

    std::string get_name()
    {
        return device_name;
    }
    const char* get_cstr_name()
    {
        return device_name.c_str();
    }
    ibv_device* get_device()
    {
        return device;
    }
    ibv_context* get_context()
    {
        return device_ctx;
    }
    ibv_device_attr get_device_attr()
    {
        return device_attr;
    }
};

class IBDevices
{
private:
    // vector container
    std::vector<IBDev_ptr_t> devices;

    // original returned object kept for destructor
    ibv_device** queried_device_list;
    int num_devices;

public:
    IBDevices()
        : queried_device_list(ibv_get_device_list(&num_devices))
    {
        // init
        devices.clear();

        if (num_devices)
        {
            for (int i = 0; i < num_devices; ++i)
            {
                devices.push_back(std::make_shared<IBDevice>(queried_device_list[i]));
            }
        }

        print_devices();
    }

    ~IBDevices()
    {
        ibv_free_device_list(queried_device_list);
    }

    void print_devices()
    {
        std::stringstream stream;
        stream << "------------------------------------------------" << std::endl
               << "Number of devices: " << num_devices << std::endl;
        for (auto& device : devices)
        {
            stream << "Name:" << device->get_name() << std::endl;
            auto device_attr = device->get_device_attr();
            auto ctx = device->get_context();
            ibv_port_attr port_attr;
            auto ret = ibv_query_port(ctx, 1, &port_attr);
            auto awidth = static_cast<int>(port_attr.active_width);
            auto aspeed = static_cast<int>(port_attr.active_speed);
            IB_ERR(!ret, "ibv_query_port", ret, errno);
            stream << "  Max MR size : " << device_attr.max_mr_size / 1024 / 1024 << " MB" << std::endl
                   << "  Max WR size : " << device_attr.max_qp_wr << std::endl
                   << "  Max CQE size: " << device_attr.max_cqe << std::endl
                   << "  smlid,lid   : " << port_attr.lid << "," << port_attr.sm_lid << std::endl
                   << "  active_width: " << awidth << std::endl
                   << "  active_speed: " << aspeed << std::endl
                   << "  Max msg size: " << port_attr.max_msg_sz << std::endl;
            stream << "  Max MTU:      "
                   << (port_attr.max_mtu == IBV_MTU_256 ? 256
                                                        : port_attr.max_mtu == IBV_MTU_512
                                  ? 512
                                  : port_attr.max_mtu == IBV_MTU_1024 ? 1024
                                                                      : port_attr.max_mtu == IBV_MTU_2048
                                          ? 2048
                                          : port_attr.max_mtu == IBV_MTU_4096 ? 4096 : -1)
                   << std::endl;
            stream << "  Active MTU:   "
                   << (port_attr.active_mtu == IBV_MTU_256 ? 256
                                                           : port_attr.active_mtu == IBV_MTU_512
                                  ? 512
                                  : port_attr.active_mtu == IBV_MTU_1024 ? 1024
                                                                         : port_attr.active_mtu == IBV_MTU_2048
                                          ? 2048
                                          : port_attr.active_mtu == IBV_MTU_4096 ? 4096 : -1)
                   << std::endl;
        }
        stream << " ------------------------------------------------" << std::endl;
        gLogVerbose << stream.str();
    }

    IBDev_ptr_t get_device(std::string myname)
    {
        if (num_devices)
        {
            if (!myname.length())
            {
                return devices[0];
            }

            for (auto& dev : devices)
            {
                if (!myname.compare(dev->get_name()))
                {
                    return dev;
                }
            }
        }
        return IBDev_ptr_t{};
    }

    std::vector<IBDev_ptr_t> get_devices()
    {
        return devices;
    }
};

class IBResources
{
private:
    // configs
    IBConfigs cfgs;

    // device
    IBDev_ptr_t device;

    // HCA device & port attribute
    ibv_device_attr device_attr;
    ibv_port_attr port_attr;

    // remote side info for data and ack
    lon_ib_con_t remote_properties;
    lon_ib_con_t remote_ack_properties;

    // device context obtained from ibv_open_device
    ibv_context* ctx;

    // queues, protection domain and memory region
    ibv_qp* qp;
    ibv_cq* send_cq;
    ibv_cq* recv_cq;
    ibv_qp* ack_qp;
    ibv_cq* send_ack_cq;
    ibv_cq* recv_ack_cq;
    ibv_pd* pd;
    ibv_mr* send_mr;
    ibv_mr* recv_mr;

    // for event notification
    void* data_send_ev_ctx;
    void* data_recv_ev_ctx;
    void* ack_send_ev_ctx;
    void* ack_recv_ev_ctx;
    ibv_comp_channel* data_send_ev_ch;
    ibv_comp_channel* data_recv_ev_ch;
    ibv_comp_channel* ack_send_ev_ch;
    ibv_comp_channel* ack_recv_ev_ch;

    // some settings
    bool LON_memory_staging;
    bool CQ_wait_event;

    // RDMA buf to exchange data
    char* send_buf;
    std::size_t send_buf_size;
    char* recv_buf;
    std::size_t recv_buf_size;

    uint8_t send_elem_num;
    std::size_t send_elem_size;
    std::size_t send_queue_size;
    uint8_t recv_elem_num;
    std::size_t recv_elem_size;
    std::size_t recv_queue_size;

    // ID for this QP
    int32_t qp_idx;

    // NUMA node
    int32_t NUMA_node;
    int32_t num_NUMA_nodes;

    // GPU
    bool on_GPU;
    int GPU_index;

    // SUT buffer may be in host memory
    bool SUT_recv_with_host_mem_for_RDMA;
    bool SUT_send_with_host_mem_for_RDMA;

public:
    // init resources using IBConfigs/IBDevices instance
    explicit IBResources(IBConfigs& cfgs_arg, IBDev_ptr_t device_arg);

    ~IBResources()
    {
        gLogVerbose << "IBResources are being destroyed: QPs, CQs, PDs, buffers and MRs..." << std::endl;
        // context shall be destroyed when Device is destroyed
        if (qp)
            ibv_destroy_qp(qp);
        dealloc_buffer_and_deregister();
        if (send_cq)
            ibv_destroy_cq(send_cq);
        if (recv_cq)
            ibv_destroy_cq(recv_cq);
        if (pd)
            ibv_dealloc_pd(pd);
    };

    void print_remote_properties()
    {
        std::stringstream stream;
        stream << std::endl
               << (cfgs.get_is_SUT() ? "SUT" : "LON") << ":" << cfgs.get_device_name() << " remote properties"
               << std::endl;
        stream << "  Remote address:        " << std::hex << remote_properties.addr << std::dec << std::endl
               << "  Remote data QP number: " << std::hex << remote_properties.qp_num << std::dec << std::endl
               << "  Remote ack  QP number: " << std::hex << remote_ack_properties.qp_num << std::dec << std::endl
               << "  Remote LID:            " << std::hex << remote_properties.lid << std::dec << std::endl
               << "  Remote rkey:           " << std::hex << remote_properties.rkey << std::dec << std::endl
               << "  Remote canary:         " << std::hex << remote_properties.canary << std::dec << std::endl
               << "------------------------------------------------" << std::endl;
        gLogVerbose << stream.str();
    }

    char* alloc_buffer(std::size_t buf_size, bool gmem);
    ibv_mr* register_memory(char* my_buf, std::size_t buf_size);
    void dealloc_buffer_and_deregister();

    IBConfigs get_configs()
    {
        return cfgs;
    }
    IBDev_ptr_t get_device()
    {
        return device;
    }
    ibv_device_attr get_device_attr()
    {
        return device_attr;
    }
    ibv_port_attr get_port_attr()
    {
        return port_attr;
    }
    ibv_context* get_context()
    {
        return ctx;
    }

    // event related
    void* get_data_send_context()
    {
        return data_send_ev_ctx;
    }
    void* get_data_recv_context()
    {
        return data_recv_ev_ctx;
    }
    void* get_ack_send_context()
    {
        return ack_send_ev_ctx;
    }
    void* get_ack_recv_context()
    {
        return ack_recv_ev_ctx;
    }
    ibv_comp_channel* get_data_send_channel()
    {
        return data_send_ev_ch;
    }
    ibv_comp_channel* get_data_recv_channel()
    {
        return data_recv_ev_ch;
    }
    ibv_comp_channel* get_ack_send_channel()
    {
        return ack_send_ev_ch;
    }
    ibv_comp_channel* get_ack_recv_channel()
    {
        return ack_recv_ev_ch;
    }

    // channel for ACK
    ibv_qp* get_ack_queue_pair()
    {
        return ack_qp;
    }
    ibv_cq* get_send_ack_completion_queue()
    {
        return send_ack_cq;
    }
    ibv_cq* get_recv_ack_completion_queue()
    {
        return recv_ack_cq;
    }

    // channel for DATA
    ibv_qp* get_queue_pair()
    {
        return qp;
    }
    ibv_cq* get_send_completion_queue()
    {
        return send_cq;
    }
    ibv_cq* get_recv_completion_queue()
    {
        return recv_cq;
    }

    ibv_pd* get_protection_domain()
    {
        return pd;
    }
    ibv_mr* get_send_memory_region()
    {
        return send_mr;
    }
    ibv_mr* get_recv_memory_region()
    {
        return recv_mr;
    }

    char* get_send_buffer()
    {
        return send_buf;
    }
    std::size_t get_send_buffer_size()
    {
        return send_buf_size;
    }
    void set_send_buffer(char* my_buf, std::size_t my_buf_size)
    {
        send_buf = my_buf;
        send_buf_size = my_buf_size;
    }
    void set_send_buffer_size(std::size_t buf_size_arg)
    {
        send_buf_size = buf_size_arg;
    }
    char* get_recv_buffer()
    {
        return recv_buf;
    }
    std::size_t get_recv_buffer_size()
    {
        return recv_buf_size;
    }
    void set_recv_buffer(char* my_buf, std::size_t my_buf_size)
    {
        recv_buf = my_buf;
        recv_buf_size = my_buf_size;
    }
    void set_recv_buffer_size(std::size_t buf_size_arg)
    {
        recv_buf_size = buf_size_arg;
    }
    lon_ib_con_t get_remote_properties()
    {
        return remote_properties;
    }
    void set_remote_properties(lon_ib_con_t properties)
    {
        remote_properties = properties;
    }
    lon_ib_con_t get_remote_ack_properties()
    {
        return remote_ack_properties;
    }
    void set_remote_ack_properties(lon_ib_con_t properties)
    {
        remote_ack_properties = properties;
    }
    uint8_t get_send_elem_num()
    {
        return send_elem_num;
    }
    uint8_t get_recv_elem_num()
    {
        return recv_elem_num;
    }
    std::size_t get_send_elem_size()
    {
        return send_elem_size;
    }
    std::size_t get_recv_elem_size()
    {
        return recv_elem_size;
    }
    std::size_t get_send_queue_size()
    {
        return send_queue_size;
    }
    std::size_t get_recv_queue_size()
    {
        return recv_queue_size;
    }
    int32_t get_NUMA_node()
    {
        return NUMA_node;
    }
    int get_gpu_idx()
    {
        return GPU_index;
    }
    int get_qp_idx()
    {
        return qp_idx;
    }
    bool get_SUT_recv_with_host_mem_for_RDMA()
    {
        return SUT_recv_with_host_mem_for_RDMA;
    }
    bool get_SUT_send_with_host_mem_for_RDMA()
    {
        return SUT_send_with_host_mem_for_RDMA;
    }
    bool get_allow_inline()
    {
        return SUT_send_with_host_mem_for_RDMA;
    }
    bool get_LON_memory_staging()
    {
        return LON_memory_staging;
    }
    bool get_CQ_wait_event()
    {
        return CQ_wait_event;
    }
};

class IBConnection
{
private:
    // resources
    IBRes_ptr_t res;

    // buffer manager for RDMA write
    IBOM_ptr_t s_om;

    bool is_SUT;
    std::string nodename;

    std::atomic<bool> continue_poll;
    mutable std::mutex poll_mutex;

    std::condition_variable data_sq_cv;
    std::condition_variable data_rq_cv;
    std::condition_variable ack_sq_cv;
    std::condition_variable ack_rq_cv;

public:
    IBConnection() {}
    IBConnection(IBRes_ptr_t res_arg, bool is_SUT_arg, std::string& nodename_arg);
    ~IBConnection() {}

    std::string opcode_str(int opcode)
    {
        std::string opc_str = "UNKNOWN";
        switch (opcode)
        {
        case IBV_WR_RDMA_WRITE: opc_str = "IBV_WR_RDMA_WRITE"; break;
        case IBV_WR_RDMA_WRITE_WITH_IMM: opc_str = "IBV_WR_RDMA_WRITE_WITH_IMM"; break;
        case IBV_WR_SEND: opc_str = "IBV_WR_SEND"; break;
        case IBV_WR_SEND_WITH_IMM: opc_str = "IBV_WR_SEND_WITH_IMM"; break;
        case IBV_WR_RDMA_READ: opc_str = "IBV_WR_RDMA_READ"; break;
        case IBV_WR_ATOMIC_CMP_AND_SWP: opc_str = "IBV_WR_ATOMIC_CMP_AND_SWP"; break;
        case IBV_WR_ATOMIC_FETCH_AND_ADD: opc_str = "IBV_WR_ATOMIC_FETCH_AND_ADD"; break;
        default: break;
        }

        return opc_str;
    }

    std::string get_nodename()
    {
        return nodename;
    }
    IBOM_ptr_t get_send_om()
    {
        return s_om;
    }

    int modify_qp_to_init(ibv_qp* qp);
    int modify_qp_to_rtr(ibv_qp* qp, lon_ib_con_t const remote_properties);
    int modify_qp_to_rts(ibv_qp* qp);
    int connect_qp(ibv_qp* qp, lon_ib_con_t const remote_properties);

    // used for data transfers
    int post_to_SQ(ibv_wr_opcode opcode, uint64_t id, uint32_t imm_data, uint32_t length = 0, uint64_t offset = 0,
        bool check_error = true);
    int post_to_RQ(
        ibv_wr_opcode opcode, uint64_t id, uint32_t length = 0, uint64_t offset = 0, bool check_error = true);

    // sq == true then poll send_cq; false poll recv_cq
    // return ID, immediate, and size
    // for data transfers
    IBWC_data_t poll_completion(bool sq, IBWC_arr_ptr_t wc_arr);

    // for ack completions; sq == true then poll send_ack_cq; false poll recv_ack_cq
    // return ID, and immediate
    std::tuple<uint64_t, uint32_t> poll_ack_completion(bool sq, IBWC_arr_ptr_t wc_arr);

    // Ack will use separate QP
    int post_send_ack(uint64_t id, uint32_t imm_data, bool check_error = true);
    int post_recv_ack(uint64_t id, bool check_error = true);

    // post null msgs for cleanup purpose
    void post_null_msgs();

    // bulk process
    std::tuple<ibv_send_wr, ibv_sge> get_send_wr(ibv_wr_opcode opcode, uint64_t id, uint32_t imm_data,
        uint32_t length = 0, uint64_t offset = 0, bool signal = true, bool solicit = false, bool ack = false,
        uint64_t override_address = 0);
    std::tuple<ibv_recv_wr, ibv_sge> get_recv_wr(
        ibv_wr_opcode opcode, uint64_t id, uint32_t length = 0, uint64_t offset = 0, bool ack = false);
    int bulk_post_to_SQ(ibv_send_wr& wr, bool ack = false, bool check_error = true);
    int bulk_post_to_RQ(ibv_recv_wr& wr, bool ack = false, bool check_error = true);
    std::unique_ptr<std::vector<IBWC_data_t>> bulk_poll_completion(bool sq, IBWC_arr_ptr_t wc_arr, bool ack = false,
        std::chrono::nanoseconds sleep_duration = std::chrono::nanoseconds(1000), int bulksize = MAX_BULK_SIZE);
    std::unique_ptr<std::vector<IBWC_data_t>> bulk_poll_completion_by_event(
        bool sq, IBWC_arr_ptr_t wc_arr, bool ack = false, bool solicit_only = false, int bulksize = MAX_BULK_SIZE);
    void req_notify_cq(bool sq, bool ack = false, bool solicit_only = false);

    bool set_continue_poll(bool val = true)
    {
        continue_poll.store(val, std::memory_order_relaxed);
        return continue_poll.load(std::memory_order_relaxed);
    }
    void clear_continue_poll()
    {
        continue_poll.store(false, std::memory_order_release);
    }

    const bool get_continue_poll() const
    {
        return continue_poll.load(std::memory_order_relaxed);
    }

    IBRes_ptr_t get_resources()
    {
        return res;
    }
};

class IBConnections
{
private:
    // for TCP initial connection establishment
    int sock;
    addrinfo* resolved_addr;

    std::string SUT_name;
    std::string LON_name;
    std::string TCP_port;

    bool is_SUT;

    // HCA devices
    std::shared_ptr<IBDevices> devices;

    // list of IBConnection
    std::vector<IBCon_ptr_t> connections;

public:
    IBConnections() {}
    IBConnections(std::vector<conn_param_t> connection_map, bool is_SUT_arg, std::string SUT_name_arg,
        std::string LON_name_arg, std::string TCP_port_arg);
    ~IBConnections()
    {
        if (sock > -1)
            close(sock);
        if (resolved_addr)
            freeaddrinfo(resolved_addr);
    }

    std::string& get_SUT_name()
    {
        return SUT_name;
    }
    std::string& get_LON_name()
    {
        return LON_name;
    }

    std::string& get_TCP_port()
    {
        return TCP_port;
    }

    int connect_tcp(const std::string& nodename, const std::string& port, addrinfo* resolved_addr, bool is_SUT);
    int sock_data_exchange(int transfer_size, char* local_data, char* remote_data);
    void sock_shutdown()
    {
        if (!is_SUT && sock > 0)
        {
            shutdown(sock, SHUT_RDWR);
            close(sock);
            sock = -1;
        }
    }

    IBCon_ptr_t get_connection(int idx)
    {
        return connections.at(idx);
    }
    std::size_t get_num_connections()
    {
        return connections.size();
    }

    int fill_remote_properties(IBRes_ptr_t res, std::string& nodename);
};

class IBOffsetManager
{
public:
    explicit IBOffsetManager(std::size_t total_item_num)
        : m_vector_size(total_item_num)
        , m_size(0)
#if TIMER_ON
        , m_total_lifetime_count(0)
        , m_total_lifetime(0)
        , m_aggregated_occupancy(0)
#endif
    {
        std::lock_guard<std::mutex> lock(om_mutex);
        m_vector.clear();
        m_vector.resize(m_vector_size, IBOMelem_t{0});
        m_deque.clear();
        for (uint32_t i = 0; i < m_vector_size; i++)
        {
            m_deque.push_back(i);
        }
        CHECK(m_vector_size > 0) << "OM should manage some transactions";
    }

    ~IBOffsetManager()
    {
        CHECK(get_occupancy() == 0) << "OM management is defective";
        CHECK_EQ(m_deque.size(), m_vector_size) << "OM management is defective";
#if TIMER_ON
        auto ave_lifetime = m_total_lifetime_count == 0 ? 0 : m_total_lifetime.count() / m_total_lifetime_count;
        gLogInfo << "OM Lifetime reports " << ave_lifetime << " ms per call for " << m_total_lifetime_count << " times."
                 << std::endl;
        auto avg_occ = m_total_lifetime_count == 0
            ? 0
            : static_cast<long double>(m_aggregated_occupancy) / static_cast<long double>(m_total_lifetime_count);
        gLogInfo << "OM average occupancy reports " << avg_occ << std::endl;
#endif
    }

    // lock and fetch available index from front of the deque
    const std::size_t get_next_available_index()
    {
        if (full())
        {
            return SIZE_MAX;
        }

        if (m_vector_size == m_size.load(std::memory_order_acquire))
            return SIZE_MAX;

        std::lock_guard<std::mutex> lock(om_mutex);

        auto idx = m_deque.front();
        m_deque.pop_front();

        m_size.store(m_vector_size - m_deque.size(), std::memory_order_release);

        return idx;
    }

    // push item in deque and return offset
    uint32_t const push(const IBOMelem_t elem)
    {
        auto idx = SIZE_MAX;
        do
        {
            std::this_thread::yield();
        } while (SIZE_MAX == (idx = get_next_available_index()));

        m_vector[idx] = elem;

#if TIMER_ON
        m_lifetime_tracker[elem] = std::chrono::high_resolution_clock::now();
#endif

        return idx;
    }

    // push item in deque and return offset
    std::vector<uint32_t> const bulk_push(std::vector<IBOMelem_t>& elems)
    {
        auto const req_size = elems.size();

        std::vector<uint32_t> indices;
        indices.reserve(req_size);

        for (auto& elem : elems)
        {
            indices.emplace_back(push(elem));
#if TIMER_ON
            m_lifetime_tracker[elem] = std::chrono::high_resolution_clock::now();
#endif
        }

        return indices;
    }

    // push item in deque by explicit index; no need to check overflow
    void push_by_index(const IBOMelem_t elem, uint32_t idx)
    {
        std::lock_guard<std::mutex> lock(om_mutex);

        // find this offset and remove it frm m_deque
        auto it = std::find(m_deque.begin(), m_deque.end(), idx);

        // error
        if (it == m_deque.end())
        {
            CHECK(false) << "Cannot find index: " << idx;
        }

        m_deque.erase(it);
        m_size.store(m_vector_size - m_deque.size(), std::memory_order_release);

        m_vector[idx] = elem;

#if TIMER_ON
        m_lifetime_tracker[elem] = std::chrono::high_resolution_clock::now();
#endif
    }

    // push item in deque by explicit index; no need to check overflow
    void bulk_push_by_indices(std::vector<std::tuple<IBOMelem_t, uint32_t>>& elems_and_indices)
    {
        CHECK(!elems_and_indices.empty()) << "Need elements and indices to push";

        std::lock_guard<std::mutex> lock(om_mutex);

        for (auto& eai : elems_and_indices)
        {
            auto& [elem, idx] = eai;

            // find this offset and remove it frm m_deque
            auto it = std::find(m_deque.begin(), m_deque.end(), idx);

            // error
            if (it == m_deque.end())
            {
                CHECK(false) << "Cannot find index: " << idx;
            }

            m_deque.erase(it);
            m_vector[idx] = elem;

#if TIMER_ON
            m_lifetime_tracker[elem] = std::chrono::high_resolution_clock::now();
#endif
        }

        m_size.store(m_vector_size - m_deque.size(), std::memory_order_release);
    }

    // pop item and return index
    uint32_t const pop_by_elem(const IBOMelem_t elem)
    {
        if (empty())
        {
            CHECK(false) << "Cannot pop as OM sees no transaction registered!";
            return 0;
        }
        else
        {
            std::lock_guard<std::mutex> lock(om_mutex);
            auto it = std::find(m_vector.begin(), m_vector.end(), elem);

            // error
            if (it == m_vector.end())
            {
                CHECK(false) << "Cannot find element: " + std::to_string(elem)
                        + ", Size: " + std::to_string(get_occupancy());
            }

            auto idx = it - m_vector.begin();
            m_deque.emplace_back(idx);

            CHECK(m_deque.size() <= m_vector_size) << "OM failed!";

            *it = IBOMelem_t{0};

#if TIMER_ON
            ++m_total_lifetime_count;
            m_total_lifetime += std::chrono::high_resolution_clock::now() - m_lifetime_tracker[elem];
            m_lifetime_tracker.erase(elem);
            m_aggregated_occupancy += get_occupancy();
#endif

            m_size.store(m_vector_size - m_deque.size(), std::memory_order_release);

            return idx;
        }
    }

    // pop item by index and return IBOMelem_t
    IBOMelem_t const pop(const uint32_t idx)
    {
        if (empty())
        {
            CHECK(false) << "Cannot pop as OM sees no transaction registered!";
        }
        else
        {
            CHECK(idx <= m_vector_size) << "Index received to pop is out of bound: " + std::to_string(idx);

            std::lock_guard<std::mutex> lock(om_mutex);
            m_deque.push_back(idx);

            CHECK(m_deque.size() <= m_vector_size) << "OM failed!";

            auto content = m_vector[idx];
            m_vector[idx] = IBOMelem_t{0};

#if TIMER_ON
            ++m_total_lifetime_count;
            m_total_lifetime += std::chrono::high_resolution_clock::now() - m_lifetime_tracker[content];
            m_lifetime_tracker.erase(content);
            m_aggregated_occupancy += get_occupancy();
#endif
            m_size.store(m_vector_size - m_deque.size(), std::memory_order_release);

            return content;
        }
        return IBOMelem_t{0};
    }

    // pop item by index and return IBOMelem_t
    std::vector<IBOMelem_t> const bulk_pop(std::vector<uint32_t>& indices)
    {
        CHECK(!indices.empty()) << "Need indices to pop";

        if (empty())
        {
            CHECK(false) << "Cannot pop as OM sees no transaction registered!";
        }
        else
        {
            std::vector<IBOMelem_t> contents;
            contents.reserve(indices.size());

            std::lock_guard<std::mutex> lock(om_mutex);

            for (auto& idx : indices)
            {

                CHECK(idx <= m_vector_size) << "Index received to pop is out of bound: " + std::to_string(idx);
                m_deque.emplace_back(idx);

                CHECK(m_deque.size() <= m_vector_size) << "OM failed!";

                auto content = m_vector[idx];
                m_vector[idx] = IBOMelem_t{0};

#if TIMER_ON
                ++m_total_lifetime_count;
                m_total_lifetime += std::chrono::high_resolution_clock::now() - m_lifetime_tracker[content];
                m_lifetime_tracker.erase(content);
                m_aggregated_occupancy += get_occupancy();
#endif
                contents.emplace_back(content);
            }

            m_size.store(m_vector_size - m_deque.size(), std::memory_order_release);

            return contents;
        }
        return std::move(std::vector<IBOMelem_t>(1, 0));
    }

    // peek item by index and return IBOMelem_t on it, without lock
    IBOMelem_t const peek(uint32_t const idx)
    {
        if (empty())
        {
            CHECK(false) << "Cannot peek as OM sees no transaction registered!";
        }
        else
        {
            CHECK(idx <= m_vector_size) << "Index to peek is out of bound: " + std::to_string(idx);

            auto content = m_vector[idx];
            CHECK_NE(content, 0) << "Invalid entry found";

            return content;
        }
        return IBOMelem_t{0};
    }

    void reset()
    {
        std::lock_guard<std::mutex> lock(om_mutex);
        m_vector.clear();
        m_vector.resize(m_vector_size, IBOMelem_t{0});
        m_deque.clear();
        m_size.store(0, std::memory_order_release);
        for (auto i = 0; i < m_vector_size; i++)
        {
            m_deque.emplace_back(i);
        }
    }

    bool const empty() const
    {
        return 0 == m_size.load(std::memory_order_relaxed);
    }

    bool const full() const
    {
        return m_vector_size == m_size.load(std::memory_order_relaxed);
    }

    std::size_t const get_occupancy() const
    {
        return m_size.load(std::memory_order_relaxed);
    }

private:
    std::size_t const m_vector_size;

    std::mutex om_mutex;

    // vector holds IBOMelem_t; unique ID cannot be zero so zero means it's invalid entry in m_vector
    // deque holds indices of m_vector whose entries are invalid
    std::vector<IBOMelem_t> m_vector;
    std::deque<std::size_t> m_deque;

    // size
    std::atomic<uint32_t> m_size;

#if TIMER_ON
    std::unordered_map<IBOMelem_t, std::chrono::time_point<std::chrono::high_resolution_clock>> m_lifetime_tracker;
    std::size_t m_total_lifetime_count;
    std::chrono::duration<double, std::milli> m_total_lifetime;
    std::size_t m_aggregated_occupancy;
#endif
};

} // namespace lonib

#endif
