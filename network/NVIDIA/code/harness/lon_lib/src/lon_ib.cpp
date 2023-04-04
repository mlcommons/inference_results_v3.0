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

#include "lon_ib.hpp"

#define USE_POLL_TIMEOUT 0
#define CQ_POLL_TIMEOUT_MS 10000
#define TCP_POLL_TIMEOUT_MS 10000
#define GPU_PAGE_SIZE 2 * 1024 * 1024
namespace lonib
{

IBResources::IBResources(IBConfigs& cfgs_arg, IBDev_ptr_t device_arg)
    : cfgs(cfgs_arg)
    , device(device_arg)
{
    int ret = 0;

    NUMA_node = cfgs.get_NUMA_node();
    num_NUMA_nodes = cfgs.get_num_NUMA_nodes();

    on_GPU = cfgs.get_is_SUT();
    GPU_index = cfgs.get_gpu_idx();
    qp_idx = cfgs.get_qp_idx();

    SUT_recv_with_host_mem_for_RDMA = cfgs.get_is_SUT() && cfgs.get_SUT_recv_with_host_mem_for_RDMA();
    SUT_send_with_host_mem_for_RDMA = cfgs.get_is_SUT() && cfgs.get_SUT_send_with_host_mem_for_RDMA();

    gLogVerbose << "Populating Device and port attributes" << std::endl;
    ctx = device->get_context();
    device_attr = device->get_device_attr();
    ret = ibv_query_port(ctx, cfgs.get_ib_port(), &port_attr);
    IB_ERR(!ret, "ibv_query_port", ret, errno);

    gLogVerbose << "Bringing up PD and MR" << std::endl;
    // protection domain
    pd = ibv_alloc_pd(ctx);
    IB_ERR(pd, "ibv_alloc_pd", ret, errno);

    send_elem_num = cfgs.get_send_elem_num();
    send_elem_size = cfgs.get_send_elem_size();
    recv_elem_num = cfgs.get_recv_elem_num();
    recv_elem_size = cfgs.get_recv_elem_size();
    send_queue_size = cfgs.get_send_queue_size();
    recv_queue_size = cfgs.get_recv_queue_size();

    data_send_ev_ch = ibv_create_comp_channel(ctx);
    IB_ERR(pd, "ibv_create_comp_channel", ret, errno);
    data_recv_ev_ch = ibv_create_comp_channel(ctx);
    IB_ERR(pd, "ibv_create_comp_channel", ret, errno);
    ack_send_ev_ch = ibv_create_comp_channel(ctx);
    IB_ERR(pd, "ibv_create_comp_channel", ret, errno);
    ack_recv_ev_ch = ibv_create_comp_channel(ctx);
    IB_ERR(pd, "ibv_create_comp_channel", ret, errno);

    LON_memory_staging = cfgs.get_LON_memory_staging();
    CQ_wait_event = cfgs.get_CQ_wait_event();

    // allocate buffer and register as MR
    // buffer for sending

    // if not staging, QSL should be loaded into registered memory, so allocate
    // large memory
    auto const send_buf_size = (!on_GPU && !LON_memory_staging)
        ? std::max(PROVISIONED_SEND_BUF_SIZE, cfgs.get_send_buffer_size())
        : cfgs.get_send_buffer_size();
    set_send_buffer_size(send_buf_size);
    gLogVerbose << "Allocating buffer and register memory region for sending" << std::endl;
    send_buf = alloc_buffer(send_buf_size, !cfgs.get_SUT_send_with_host_mem_for_RDMA());
    send_mr = register_memory(send_buf, send_buf_size);
    CHECK(send_buf != nullptr & send_mr != nullptr) << "Send buffer allocation and/or its memory registration failed";
    // buffer for receiving
    auto const recv_buf_size = cfgs.get_recv_buffer_size();
    set_recv_buffer_size(recv_buf_size);
    gLogVerbose << "Allocating buffer and register memory region for receiving" << std::endl;
    recv_buf = alloc_buffer(recv_buf_size, !cfgs.get_SUT_recv_with_host_mem_for_RDMA());
    recv_mr = register_memory(recv_buf, recv_buf_size);
    CHECK(recv_buf != nullptr & recv_mr != nullptr) << "Recv buffer allocation and/or its memory registration failed";

    gLogVerbose << "Bringing up CQ and QP" << std::endl;
    // completion queue for data
    send_cq = ibv_create_cq(ctx, static_cast<int>(send_queue_size), data_send_ev_ctx, data_send_ev_ch, 0);
    IB_ERR(send_cq, "ibv_create_cq", ret, errno);
    recv_cq = ibv_create_cq(ctx, static_cast<int>(recv_queue_size), data_recv_ev_ctx, data_recv_ev_ch, 0);
    IB_ERR(recv_cq, "ibv_create_cq", ret, errno);

    // queue pair for data; max queue size have slack, i.e. 10+ to expected trx on
    // the fly
    // adding slack to minimize retry during queue size transition
    ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.send_cq = send_cq;
    qp_init_attr.recv_cq = recv_cq;
    CHECK_LE(send_queue_size, MLX_MAX_WR);
    CHECK_LE(recv_queue_size, MLX_MAX_WR);
    qp_init_attr.cap.max_send_wr = send_queue_size;
    qp_init_attr.cap.max_recv_wr = recv_queue_size;
    qp_init_attr.cap.max_inline_data = INLINE_SIZE_THRESHOLD;
    // Let's not use complicated scatter/gather request; fix max sge to 1
    qp_init_attr.cap.max_send_sge = qp_init_attr.cap.max_recv_sge = 1;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.sq_sig_all = 0;
    qp = ibv_create_qp(get_protection_domain(), &qp_init_attr);
    IB_ERR(qp, "ibv_create_qp", -1, errno);
    gLogVerbose << "DATA QP[" << std::hex << qp->qp_num << std::dec << "] successfully created" << std::endl;

    // completion queue for ack; queue size follows data side (opposite dir)
    send_ack_cq = ibv_create_cq(ctx, static_cast<int>(recv_queue_size), ack_send_ev_ctx, ack_send_ev_ch, 0);
    IB_ERR(send_ack_cq, "ibv_create_cq", ret, errno);
    recv_ack_cq = ibv_create_cq(ctx, static_cast<int>(send_queue_size), ack_recv_ev_ctx, ack_recv_ev_ch, 0);
    IB_ERR(recv_ack_cq, "ibv_create_cq", ret, errno);

    // queue pair for ack; queue size follows data side (opposite dir)
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.send_cq = send_ack_cq;
    qp_init_attr.recv_cq = recv_ack_cq;
    qp_init_attr.cap.max_send_wr = send_queue_size;
    qp_init_attr.cap.max_recv_wr = recv_queue_size;
    qp_init_attr.cap.max_inline_data = INLINE_SIZE_THRESHOLD;
    qp_init_attr.cap.max_send_sge = qp_init_attr.cap.max_recv_sge = 1;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.sq_sig_all = 0;
    ack_qp = ibv_create_qp(get_protection_domain(), &qp_init_attr);
    IB_ERR(ack_qp, "ibv_create_qp", -1, errno);
    gLogVerbose << "ACK QP[" << std::hex << ack_qp->qp_num << std::dec << "] successfully created" << std::endl;
}

char* IBResources::alloc_buffer(std::size_t buf_size, bool gmem)
{
    int ret = 0;
    char* my_buf = nullptr;

    gLogVerbose << "Allocating buffers for exchanging data" << std::endl;

    // allocate memory buf for exchanging data
    if (on_GPU && gmem)
    {
        cudaError_t cudaerr;
        cudaerr = cudaSetDevice(GPU_index);
        CHECK(cudaerr == cudaSuccess) << "Failed to set GPU device of index " + GPU_index;
        cudaerr = cudaMalloc(reinterpret_cast<void**>(&my_buf), buf_size);
        CHECK(cudaerr == cudaSuccess) << "Failed to allocate GPU memory of size " + std::to_string(buf_size) + "B";
        gLogVerbose << "GPU[" << GPU_index << "] buffer allocated at addr=" << std::hex << static_cast<void*>(my_buf)
                    << std::dec << std::endl;
    }
    else
    {
        // if NUMA node is set, set mem bind policy
        if (num_NUMA_nodes > 1 && NUMA_node >= 0)
        {
            bindNumaMemPolicy(NUMA_node, num_NUMA_nodes);
        }

        if (on_GPU)
        {
            my_buf = static_cast<char*>(malloc(buf_size));

            // Pin the memory
            // NOTE: it seems that cudaHostAlloc/cudaMallocHost would NOT support NUMA
            // membind
            // NOTE: buf_size is already aligned to 2M page for GPU memory so should
            // align with host memory page as well
            cudaError_t cudaerr;
            cudaerr = cudaSetDevice(GPU_index);
            CHECK(cudaerr == cudaSuccess) << "Failed to set GPU device of index " + GPU_index;
            cudaerr = cudaHostRegister(reinterpret_cast<void*>(my_buf), buf_size, cudaHostAllocDefault);
            CHECK(cudaerr == cudaSuccess)
                << "Failed to allocate/pin Host memory of size " + std::to_string(buf_size) + "B";
            gLogVerbose << "Host buffer allocated/pinned at addr=" << std::hex << static_cast<void*>(my_buf) << std::dec
                        << " NUMA[" << NUMA_node << "] / GPU[" << GPU_index << "]" << std::endl;
        }
        else
        {
            my_buf = static_cast<char*>(malloc(buf_size));
            CHECK(my_buf != nullptr) << "Failed to allocate host memory of size " + std::to_string(buf_size) + "B";
            gLogVerbose << "Host buffer allocated/pinned at addr=" << std::hex << static_cast<void*>(my_buf) << std::dec
                        << " NUMA[" << NUMA_node << "]" << std::endl;
        }

        // if NUMA node is set, reset mem bind policy
        if (num_NUMA_nodes > 1 && NUMA_node >= 0)
        {
            resetNumaMemPolicy();
        }
    }
    return my_buf;
}

ibv_mr* IBResources::register_memory(char* my_buf, std::size_t buf_size)
{
    // memory region
    ibv_access_flags access
        = static_cast<ibv_access_flags>(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

    auto my_mr = ibv_reg_mr(get_protection_domain(), my_buf, buf_size, access);
    IB_ERR(my_mr != nullptr, "ibv_reg_mr", -1, errno);

    gLogVerbose << "MR registered with buf addr=" << std::hex << static_cast<void*>(my_buf) << ", lkey=" << my_mr->lkey
                << ", rkey=" << my_mr->rkey << ", flags=" << access << std::dec << ", size=" << buf_size << std::endl;

    return my_mr;
}

void IBResources::dealloc_buffer_and_deregister()
{
    // deregister memory region
    gLogVerbose << "Deallocating buffers and Deregistering memory regions" << std::endl;
    if (send_mr)
        ibv_dereg_mr(send_mr);
    if (recv_mr)
        ibv_dereg_mr(recv_mr);
    if (send_buf)
    {
        if (on_GPU && !SUT_send_with_host_mem_for_RDMA)
        {
            cudaSetDevice(GPU_index);
            cudaFree(static_cast<void*>(send_buf));
        }
        else
        {
            if (on_GPU)
            {
                cudaHostUnregister(static_cast<void*>(send_buf));
            }
            free(static_cast<void*>(send_buf));
        }
    }
    if (recv_buf)
    {
        if (on_GPU && !SUT_recv_with_host_mem_for_RDMA)
        {
            cudaSetDevice(GPU_index);
            cudaFree(static_cast<void*>(recv_buf));
        }
        else
        {
            if (on_GPU)
            {
                cudaHostUnregister(static_cast<void*>(recv_buf));
            }
            free(static_cast<void*>(recv_buf));
        }
    }
}

IBConnection::IBConnection(IBRes_ptr_t res_arg, bool is_SUT_arg, std::string& nodename_arg)
    : res(res_arg)
    , is_SUT(is_SUT_arg)
    , nodename(nodename_arg)
    , continue_poll(false)
{
    int ret = -1;

    // connect QP
    ret = connect_qp(res->get_queue_pair(), res->get_remote_properties());
    CHECK_EQ(ret, 0);

    ret = connect_qp(res->get_ack_queue_pair(), res->get_remote_ack_properties());
    CHECK_EQ(ret, 0);

    auto cfg = res->get_configs();
    s_om = std::make_unique<IBOffsetManager>(cfg.get_send_queue_size());
}

int IBConnection::modify_qp_to_init(ibv_qp* my_qp)
{
    ibv_qp_attr attr;
    ibv_qp_attr_mask mask;

    int ret = 0;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = res->get_configs().get_ib_port();
    attr.qp_access_flags
        = static_cast<ibv_access_flags>(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

    mask = static_cast<ibv_qp_attr_mask>(IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS);

    ret = ibv_modify_qp(my_qp, &attr, mask);
    CHECK(!ret) << "Failed to move QP state to INIT";

    return ret;
}

int IBConnection::modify_qp_to_rtr(ibv_qp* my_qp, lon_ib_con_t const remote_properties)
{
    ibv_qp_attr attr;
    ibv_qp_attr_mask mask;

    int ret = 0;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote_properties.qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = remote_properties.lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = res->get_configs().get_ib_port();

    mask = static_cast<ibv_qp_attr_mask>(IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN
        | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);

    ret = ibv_modify_qp(my_qp, &attr, mask);
    CHECK(!ret) << "Failed to move QP state to RTR";

    return ret;
}

int IBConnection::modify_qp_to_rts(ibv_qp* my_qp)
{
    ibv_qp_attr attr;
    ibv_qp_attr_mask mask;

    int ret = 0;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 31;  // Maybe something like 14
    attr.retry_cnt = 7; // 7 for max; Maybe something like 3
    attr.rnr_retry = 7; // 7 for infinite retry; maybe change this to something like 3
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;

    mask = static_cast<ibv_qp_attr_mask>(
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);

    ret = ibv_modify_qp(my_qp, &attr, mask);
    CHECK(!ret) << "Failed to move QP state to RTS";

    return ret;
}

int IBConnection::connect_qp(ibv_qp* my_qp, lon_ib_con_t const remote_properties)
{
    int ret = 0;

    // init QP
    ret = modify_qp_to_init(my_qp);
    IB_ERR(!ret, "modify_qp_to_init", ret, errno);

    // modify the QP to RTR
    ret = modify_qp_to_rtr(my_qp, remote_properties);
    IB_ERR(!ret, "modify_qp_to_rtr", ret, errno);

    // move QP to RTS
    ret = modify_qp_to_rts(my_qp);
    IB_ERR(!ret, "modify_qp_to_rts", ret, errno);

    gLogVerbose << "QP[" << std::hex << my_qp->qp_num << std::dec << "] is now in RTS state" << std::endl;

    return ret;
}

int IBConnection::post_to_SQ(
    ibv_wr_opcode opcode, uint64_t id, uint32_t imm_data, uint32_t length, uint64_t offset, bool check_error)
{
    auto tuple = get_send_wr(opcode, id, imm_data, length, offset, true);
    auto& [wr, sge] = tuple;
    wr.sg_list = &sge;
    return bulk_post_to_SQ(wr, false, check_error);
}

int IBConnection::post_to_RQ(ibv_wr_opcode opcode, uint64_t id, uint32_t length, uint64_t offset, bool check_error)
{
    auto tuple = get_recv_wr(opcode, id, length, offset);
    auto& [wr, sge] = tuple;
    wr.sg_list = &sge;
    return bulk_post_to_RQ(wr, false, check_error);
}

IBWC_data_t IBConnection::poll_completion(bool sq, IBWC_arr_ptr_t wc_arr)
{
    auto cmpl = bulk_poll_completion(sq, wc_arr, false, std::chrono::microseconds(1), 1);
    return cmpl->front();
}

// ack will use 0-size transfer with immediate field
// NOTE: current design uses offset for immediate field, if anything > uint32_t
// is needed, maybe use IBV_SEND_INLINE
//       this ack is used to clean up the offset manager and is purely CPU side
// work
int IBConnection::post_send_ack(uint64_t id, uint32_t imm_data, bool check_error)
{
    auto opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    auto tuple = get_send_wr(opcode, id, imm_data, 0, 0, true, true, true);
    auto& [wr, sge] = tuple;
    wr.sg_list = &sge;
    return bulk_post_to_SQ(wr, true, check_error);
}

int IBConnection::post_recv_ack(uint64_t id, bool check_error)
{
    auto opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    auto tuple = get_recv_wr(opcode, id, 0, 0, true);
    auto& [wr, sge] = tuple;
    wr.sg_list = &sge;
    return bulk_post_to_RQ(wr, true, check_error);
}

// post null msgs (zero size msg with ID==UINT64_MAX) for cleanup purpose
void IBConnection::post_null_msgs()
{
    auto opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    post_to_SQ(opcode, UINT64_MAX, UINT32_MAX, 0, 0, false);
    post_to_RQ(opcode, UINT64_MAX, 0, 0, false);
    post_send_ack(UINT64_MAX, UINT32_MAX, false);
    post_recv_ack(UINT64_MAX, false);
}

std::tuple<uint64_t, uint32_t> IBConnection::poll_ack_completion(bool sq, IBWC_arr_ptr_t wc_arr)
{
    auto cmpl = bulk_poll_completion(sq, wc_arr, true, std::chrono::microseconds(1), 1);
    auto& [id, ret, tr_size] = cmpl->front();
    return std::make_tuple(id, ret);
}

std::tuple<ibv_send_wr, ibv_sge> IBConnection::get_send_wr(ibv_wr_opcode opcode, uint64_t id, uint32_t imm_data,
    uint32_t length, uint64_t offset, bool signal, bool solicit, bool ack, uint64_t override_address)
{
    ibv_send_wr wr;
    ibv_sge sge;

    std::string opc_str = opcode_str(opcode);
    std::stringstream ss;
    ss << std::hex << imm_data;

    auto addr = override_address > 0 ? override_address
                                     : ack ? 0 : reinterpret_cast<uint64_t>(res->get_send_buffer() + offset);

    auto lkey = res->get_send_memory_region()->lkey;

    auto allow_inline = res->get_allow_inline();

    // scatter-gather entry
    memset(&sge, 0, sizeof(ibv_sge));
    sge.addr = addr;
    sge.length = ack
        ? 0
        : length > 0 ? length : (res->get_configs().get_send_elem_size() * res->get_configs().get_send_elem_num());
    sge.lkey = lkey;

    // send work request
    auto use_inline = allow_inline && sge.length <= INLINE_SIZE_THRESHOLD && sge.length > 0;
    auto send_flag = 0 | (signal ? IBV_SEND_SIGNALED : 0) | (use_inline ? IBV_SEND_INLINE : 0)
        | (solicit ? IBV_SEND_SOLICITED : 0);

    memset(&wr, 0, sizeof(ibv_send_wr));
    wr.next = nullptr;
    wr.wr_id = id;
    wr.sg_list = &sge;
    wr.num_sge = length == 0 ? 0 : 1; // zero length message is done by num_sge==0
                                      // not length=0 in fact
    wr.opcode = ack ? IBV_WR_RDMA_WRITE_WITH_IMM : opcode;
    wr.send_flags = send_flag;

    auto remote_addr = ack ? 0 : reinterpret_cast<uint64_t>(res->get_remote_properties().addr + offset);
    auto rkey = ack ? 0 : res->get_remote_properties().rkey;

    switch (opcode)
    {
    case IBV_WR_SEND_WITH_IMM: wr.imm_data = htonl(imm_data); opc_str = opc_str + "[" + ss.str() + "]";
    case IBV_WR_SEND: break;
    case IBV_WR_RDMA_WRITE_WITH_IMM: wr.imm_data = htonl(imm_data); opc_str = opc_str + "[" + ss.str() + "]";
    case IBV_WR_RDMA_WRITE:
    case IBV_WR_RDMA_READ:
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = rkey;
        break;
    default: CHECK(false) << "Failed: unsupported opcode: " + opc_str; return std::make_tuple(wr, sge);
    }

    DLOG_IF(INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
        << "Created Send WR(" << (ack ? "ACK" : "DATA") << "): " << opc_str << ", WR_ID: " << std::hex << id << "; "
        << " Remote Base Addr: " << static_cast<uint64_t>(res->get_remote_properties().addr) << " offset: " << offset
        << " RDMA.remote_addr: " << remote_addr << " RDMA.rkey: " << wr.wr.rdma.rkey << " SGE.addr: " << sge.addr
        << std::dec << " SGE.len: " << sge.length << " SIGNAL: " << signal << " INLINE: " << use_inline << std::endl;

    return std::make_tuple(wr, sge);
}

std::tuple<ibv_recv_wr, ibv_sge> IBConnection::get_recv_wr(
    ibv_wr_opcode opcode, uint64_t id, uint32_t length, uint64_t offset, bool ack)
{
    ibv_recv_wr wr;
    ibv_sge sge;

    std::string opc_str = opcode_str(opcode);

    switch (opcode)
    {
    case IBV_WR_SEND:
    case IBV_WR_SEND_WITH_IMM:
    case IBV_WR_RDMA_WRITE_WITH_IMM: break;
    default: CHECK(false) << "Failed: unsupported opcode: " + opc_str; return std::make_tuple(wr, sge);
    }

    auto addr = ack ? 0 : reinterpret_cast<uint64_t>(res->get_recv_buffer() + offset);

    auto lkey = res->get_recv_memory_region()->lkey;

    // scatter-gather entry
    memset(&sge, 0, sizeof(ibv_sge));
    sge.addr = addr;
    sge.length = ack
        ? 0
        : length > 0 ? length : (res->get_configs().get_recv_elem_size() * res->get_configs().get_recv_elem_num());
    sge.lkey = lkey;

    // recv work request
    memset(&wr, 0, sizeof(ibv_recv_wr));
    wr.next = nullptr;
    wr.wr_id = id;
    wr.sg_list = &sge;
    wr.num_sge = length == 0 ? 0 : 1;

    DLOG_IF(INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
        << "Created Recv WR(" << (ack ? "ACK" : "DATA") << "): " << opc_str << ", WR_ID: " << std::hex << id << "; "
        << " Receive buffer base addr: " << static_cast<void*>(res->get_recv_buffer()) << " offset: " << offset
        << " RDMA.local_addr: " << sge.addr << " RDMA.lkey: " << sge.lkey << std::dec << " SGE.len: " << sge.length
        << std::endl;

    return std::make_tuple(wr, sge);
}

int IBConnection::bulk_post_to_SQ(ibv_send_wr& wr, bool ack, bool check_error)
{
    int ret = 0;
    ibv_send_wr* bad_wr = nullptr;

    auto qp = ack ? res->get_ack_queue_pair() : res->get_queue_pair();
    // post WR to SQ
    ret = ibv_post_send(qp, &wr, &bad_wr);
    if (check_error)
    {
        IB_ERR(!ret, "ibv_post_send", ret, ret);
    }

    DLOG_IF(INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
        << (ack ? "ACK" : "DATA") << " -- Bulk posting Send WRs completed" << std::endl;
    return ret;
}

int IBConnection::bulk_post_to_RQ(ibv_recv_wr& wr, bool ack, bool check_error)
{
    int ret = 0;
    ibv_recv_wr* bad_wr = nullptr;

    auto qp = ack ? res->get_ack_queue_pair() : res->get_queue_pair();
    // post WR to SQ
    ret = ibv_post_recv(qp, &wr, &bad_wr);
    if (check_error)
    {
        IB_ERR(!ret, "ibv_post_recv", ret, ret);
    }

    DLOG_IF(INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
        << (ack ? "ACK" : "DATA") << " -- Bulk posting Recv WRs completed" << std::endl;
    return ret;
}

std::unique_ptr<std::vector<IBWC_data_t>> IBConnection::bulk_poll_completion(
    bool sq, IBWC_arr_ptr_t wc_arr, bool ack, std::chrono::nanoseconds sleep_duration, int bulksize)
{
    uint64_t start_time_msec, current_time_msec;

    int poll_result = 0;
    uint64_t ret = UINT64_MAX;

    CHECK_LE(bulksize, wc_arr->size()) << "bulk size cannot be larger than the container";

    auto data_cq = sq ? res->get_send_completion_queue() : res->get_recv_completion_queue();
    auto ack_cq = sq ? res->get_send_ack_completion_queue() : res->get_recv_ack_completion_queue();
    auto cq = ack ? ack_cq : data_cq;

    auto qsize = sq ? res->get_send_queue_size() : res->get_recv_queue_size();
    auto my_bulksize = std::min(static_cast<std::size_t>(bulksize), qsize);

    // poll, with timeout
    bool timeout = false;
    start_time_msec = current_time_msec = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch())
                                              .count();
    while ((poll_result == 0) && get_continue_poll() && !timeout)
    {
        poll_result = ibv_poll_cq(cq, my_bulksize, wc_arr->data());
        current_time_msec = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
                                .count();
        timeout = USE_POLL_TIMEOUT && (current_time_msec - start_time_msec) > CQ_POLL_TIMEOUT_MS;
        if (poll_result == 0)
        {
            std::this_thread::sleep_for(sleep_duration);
        }
    }

    auto cmpl_data = std::make_unique<std::vector<IBWC_data_t>>();
    cmpl_data->clear();

    IB_ERR(poll_result != -1, "ibv_poll_cq", ret, errno);
    CHECK(!timeout) << "Failure: Poll CQ timeout";

    for (int i = 0; i < poll_result; ++i)
    {
        auto wc = wc_arr->at(i);
        CHECK(wc.status == IBV_WC_SUCCESS)
            << "Bad completion found, status: " + std::string(ibv_wc_status_str(wc.status));

        if (wc.wc_flags && IBV_WC_WITH_IMM)
        {
            ret = static_cast<uint64_t>(ntohl(wc.imm_data));
        }
        auto tr_size = wc.byte_len;
        auto id = wc.wr_id;

        DLOG_IF(INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
            << "Polling completion done for " << (ack ? "ACK" : "DATA") << " -- " << (sq ? "SQ" : "RQ")
            << ", STATUS: " << std::string(ibv_wc_status_str(wc.status)) << std::hex << ", WR_ID: " << id
            << ", OP: " << opcode_str(wc.opcode) << ", IMM: " << std::hex << ret << std::dec
            << ", SIZE: " << wc.byte_len << ", #QP: " << wc.qp_num << ", SRC_QP: " << wc.src_qp << std::dec
            << std::endl;

        cmpl_data->emplace_back(id, ret, tr_size);
    }

    if (!get_continue_poll())
    {
        cmpl_data->emplace_back(UINT64_MAX, 0, 0);
    }

    return cmpl_data;
}

std::unique_ptr<std::vector<IBWC_data_t>> IBConnection::bulk_poll_completion_by_event(
    bool sq, IBWC_arr_ptr_t wc_arr, bool ack, bool solicit_only, int bulksize)
{
    int poll_result = 0;
    uint64_t ret = 0;

    CHECK_LE(bulksize, wc_arr->size()) << "bulk size cannot be larger than the container";

    auto data_cq = sq ? res->get_send_completion_queue() : res->get_recv_completion_queue();
    auto ack_cq = sq ? res->get_send_ack_completion_queue() : res->get_recv_ack_completion_queue();
    auto cq = ack ? ack_cq : data_cq;

    auto data_ev_ctx = sq ? res->get_data_send_context() : res->get_data_recv_context();
    auto ack_ev_ctx = sq ? res->get_ack_send_context() : res->get_ack_recv_context();
    auto ev_ctx = ack ? ack_ev_ctx : data_ev_ctx;

    auto data_ev_ch = sq ? res->get_data_send_channel() : res->get_data_recv_channel();
    auto ack_ev_ch = sq ? res->get_ack_send_channel() : res->get_ack_recv_channel();
    auto ev_ch = ack ? ack_ev_ch : data_ev_ch;

    auto qsize = sq ? res->get_send_queue_size() : res->get_recv_queue_size();
    auto my_bulksize = std::min(static_cast<std::size_t>(bulksize), qsize);

    auto cmpl_data = std::make_unique<std::vector<IBWC_data_t>>();
    cmpl_data->clear();

    while (true)
    {
        poll_result = ibv_poll_cq(cq, my_bulksize, wc_arr->data());
        IB_ERR(poll_result != -1, "ibv_poll_cq", ret, errno);
        if (poll_result > 0)
        {
            break;
        }

        ret = ibv_req_notify_cq(cq, solicit_only);
        IB_ERR(!ret, "ibv_req_notify_cq", ret, ret);

        poll_result = ibv_poll_cq(cq, my_bulksize, wc_arr->data());
        IB_ERR(poll_result != -1, "ibv_poll_cq", ret, errno);
        if (poll_result > 0)
        {
            break;
        }

        ret = ibv_get_cq_event(ev_ch, &cq, &ev_ctx);
        IB_ERR(!ret, "ibv_get_cq_event", ret, errno);

        ibv_ack_cq_events(cq, 1); // notify only one event; FIXME: how to ack more than one?

        poll_result = ibv_poll_cq(cq, my_bulksize, wc_arr->data());
        IB_ERR(poll_result != -1, "ibv_poll_cq", ret, errno);

        // break out regardless of poll_result as event happened
        break;
    }

    for (int i = 0; i < poll_result; ++i)
    {
        auto wc = wc_arr->at(i);
        CHECK(wc.status == IBV_WC_SUCCESS)
            << "Bad completion found, status: " + std::string(ibv_wc_status_str(wc.status));

        if (wc.wc_flags && IBV_WC_WITH_IMM)
        {
            ret = static_cast<uint64_t>(ntohl(wc.imm_data));
        }
        auto tr_size = wc.byte_len;
        auto id = wc.wr_id;

        DLOG_IF(INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
            << "Polling completion done for " << (ack ? "ACK" : "DATA") << " -- " << (sq ? "SQ" : "RQ")
            << ", STATUS: " << std::string(ibv_wc_status_str(wc.status)) << std::hex << ", WR_ID: " << id
            << ", OP: " << opcode_str(wc.opcode) << ", IMM: " << std::hex << ret << std::dec
            << ", SIZE: " << wc.byte_len << ", #QP: " << wc.qp_num << ", SRC_QP: " << wc.src_qp << std::dec
            << std::endl;

        cmpl_data->emplace_back(id, ret, tr_size);
    }

    // if there's event that wasn't cleared but all the completions are drained,
    // this can return empty vector
    return cmpl_data;
}

void IBConnection::req_notify_cq(bool sq, bool ack, bool solicit_only)
{
    auto data_cq = sq ? res->get_send_completion_queue() : res->get_recv_completion_queue();
    auto ack_cq = sq ? res->get_send_ack_completion_queue() : res->get_recv_ack_completion_queue();
    auto cq = ack ? ack_cq : data_cq;

    auto data_ev_ctx = sq ? res->get_data_send_context() : res->get_data_recv_context();
    auto ack_ev_ctx = sq ? res->get_ack_send_context() : res->get_ack_recv_context();
    auto ev_ctx = ack ? ack_ev_ctx : data_ev_ctx;

    auto data_ev_ch = sq ? res->get_data_send_channel() : res->get_data_recv_channel();
    auto ack_ev_ch = sq ? res->get_ack_send_channel() : res->get_ack_recv_channel();
    auto ev_ch = ack ? ack_ev_ch : data_ev_ch;

    auto ret = ibv_req_notify_cq(cq, solicit_only);
    IB_ERR(!ret, "ibv_req_notify_cq", ret, ret);
}

IBConnections::IBConnections(std::vector<conn_param_t> connection_map, bool is_SUT_arg, std::string SUT_name_arg = "",
    std::string LON_name_arg = "", std::string TCP_port_arg = "35746")
    : resolved_addr(nullptr)
    , sock(-1)
    , SUT_name(SUT_name_arg)
    , LON_name(LON_name_arg)
    , is_SUT(is_SUT_arg)
    , TCP_port(TCP_port_arg)
    , devices(std::make_shared<IBDevices>())
{
    int ret = 0;

    connections.clear();

    if (is_SUT)
    {
        gLogInfo << "Waiting LON[" << LON_name << "], port " << TCP_port << " for establishing TCP connection"
                 << std::endl;
        sock = connect_tcp("localhost", TCP_port, resolved_addr, is_SUT);
        CHECK(sock >= 0) << "ERROR: failed to establish TCP connection with LON[" + LON_name + "], port " + TCP_port;
    }
    else
    {
        sock = connect_tcp(SUT_name, TCP_port, resolved_addr, is_SUT);
        CHECK(sock >= 0) << "ERROR: failed to stablish TCP connection with SUT[" + SUT_name + "], port " + TCP_port;
    }
    gLogInfo << "TCP connection established" << std::endl;

    // sanity check if data can be exchanged through connection
    uint32_t sanity_check_0{0x00000000};
    uint32_t sanity_check_1{0xDEADBEEF};
    uint32_t sanity_check_2 = htonl(sanity_check_1);
    ret = sock_data_exchange(4, reinterpret_cast<char*>(&sanity_check_2), reinterpret_cast<char*>(&sanity_check_0));
    CHECK(ret >= 0) << "TCP data exchange failed, ret=" + ret;
    CHECK(ntohl(sanity_check_0) == sanity_check_1) << "Sanity check on data exchange failed";

    std::string nodename
        = is_SUT ? (SUT_name.length() > 0 ? SUT_name : "SUT") : (LON_name.length() > 0 ? LON_name : "LON");
    for (const auto& con : connection_map)
    {
        // con[0] (LON NIC_device/neighbor_GPU) <=> con[1] (SUT
        // NIC_device/neighbor_GPU)
        // con[2]/con[3] whether SUT will use host memory for RDMA transactions, for
        // recv and send
        // con[4] if LON will stage memory to be used for RDMA transfer
        // con[5] if CQ will be maintained by blocking wait events, instead of
        // polling
        // con[6] NUMA_node and con[6] total number of NUMA nodes
        // con[7] number of QPs per NIC
        // con[8] send queue size, con[9] number of send element(s), con[10] send
        // element (max) size
        // con[11] recv queue size, con[12] number of recv element(s), con[13] recv
        // element (max) size
        // LON side, neighbor_GPU maybe N/A (-1)
        auto [lon_info, sut_info, SUT_recv_with_host_mem_for_RDMA, SUT_send_with_host_mem_for_RDMA, LON_memory_staging,
            CQ_wait_event, numa_node, num_numa, qp_idx, send_queue_size, send_elem_num, send_elem_size, recv_queue_size,
            recv_elem_num, recv_elem_size]
            = con;
        std::string device_name = is_SUT ? sut_info.first : lon_info.first;
        int gpu_index = is_SUT ? sut_info.second : lon_info.second;
        IBDev_ptr_t my_dev = devices->get_device(device_name);
        // FIXME: double check port == 1 for single port HSA

        // if NUMA node is set, set mem bind policy
        if (num_numa > 1 && numa_node >= 0)
        {
            bindNumaMemPolicy(numa_node, num_numa);
        }

        // buffer will be allocated to cover queue size, with padding of 1 entry for
        // safety
        std::size_t send_buf_size = static_cast<std::size_t>(send_queue_size * send_elem_size * send_elem_num);
        std::size_t recv_buf_size = static_cast<std::size_t>(recv_queue_size * recv_elem_size * recv_elem_num);
        std::size_t gpu_page_size{GPU_PAGE_SIZE};
        // align (div-up) buf size to GPU page size; also align CPU side as well for
        // convenience
        std::size_t aligned_send_buf_size = ((send_buf_size + gpu_page_size - 1) / gpu_page_size) * gpu_page_size;
        std::size_t aligned_recv_buf_size = ((recv_buf_size + gpu_page_size - 1) / gpu_page_size) * gpu_page_size;

        // build config
        IBConfigs my_cfg{
            device_name, is_SUT,
            SUT_recv_with_host_mem_for_RDMA, // if SUT uses host mem for recv RDMA
                                             // transfer
            SUT_send_with_host_mem_for_RDMA, // if SUT uses host mem for send RDMA
                                             // transfer
            LON_memory_staging,              // if LON needs staging for RDMA transfer
            CQ_wait_event,                   // CQ uses blocking events rather than polling
            1,                               // port 1 is used for single port HSA
            gpu_index,                       // associated GPU's idx
            qp_idx,                          // QP idx among QPs in the same NIC
            numa_node,                       // NUMA node to bind
            num_numa,                        // total number of NUMA nodes
            send_queue_size,                 // send_queue_size (number of entries)
            send_elem_num,                   // send_elem_num
            send_elem_size,                  // send_elem_size
            aligned_send_buf_size,           // send_buf_size
            recv_queue_size,                 // recv_queue_size (number of entries)
            recv_elem_num,                   // recv_elem_num
            recv_elem_size,                  // recv_elem_size
            aligned_recv_buf_size,           // recv_buf_size
        };
        my_cfg.print_configs();
        IBRes_ptr_t my_res = std::make_shared<IBResources>(my_cfg, my_dev);
        // should work as connection is established one by one in order
        ret = fill_remote_properties(my_res, nodename);
        CHECK_NE(ret, 0);
        IBCon_ptr_t my_con = std::make_shared<IBConnection>(my_res, is_SUT, device_name);
        connections.push_back(my_con);

        // if NUMA node is set, set mem bind policy
        if (num_numa > 1 && numa_node >= 0)
        {
            resetNumaMemPolicy();
        }
    }
}

int IBConnections::fill_remote_properties(IBRes_ptr_t res, std::string& nodename)
{
    int ret = 0;

    lon_ib_con_t local_con_prop = {0, 0, 0, 0, htons(0xdead)};
    lon_ib_con_t temp_con_prop = {0, 0, 0, 0, 0};
    lon_ib_con_t remote_con_prop = {0, 0, 0, 0, 0};

    // exchange using TCP sockets info required to connect QP
    local_con_prop.addr = htonll(reinterpret_cast<uintptr_t>(res->get_recv_buffer()));
    local_con_prop.rkey = htonl(res->get_recv_memory_region()->rkey);
    local_con_prop.qp_num = htonl(res->get_queue_pair()->qp_num);
    local_con_prop.lid = htons(res->get_port_attr().lid);

    gLogVerbose << nodename << ":" << res->get_device()->get_name() << ", LID=" << std::hex << res->get_port_attr().lid
                << std::dec << ", fetching remote properties for data" << std::endl;
    ret = sock_data_exchange(
        sizeof(lon_ib_con_t), reinterpret_cast<char*>(&local_con_prop), reinterpret_cast<char*>(&temp_con_prop));
    CHECK(ret >= 0) << "Failed to exchange connection data while filling remote "
                       "properties for data: "
            + nodename + "/" + res->get_device()->get_name();

    remote_con_prop.addr = ntohll(temp_con_prop.addr);
    remote_con_prop.rkey = ntohl(temp_con_prop.rkey);
    remote_con_prop.qp_num = ntohl(temp_con_prop.qp_num);
    remote_con_prop.lid = ntohs(temp_con_prop.lid);
    remote_con_prop.canary = ntohs(temp_con_prop.canary);

    // save the remote side attributes, we will need it for the post SR
    res->set_remote_properties(remote_con_prop);

    // do one more for ack QP
    local_con_prop.qp_num = htonl(res->get_ack_queue_pair()->qp_num);

    gLogVerbose << nodename << ":" << res->get_device()->get_name() << ", LID=" << std::hex << res->get_port_attr().lid
                << std::dec << ", fetching remote properties for ack" << std::endl;
    ret = sock_data_exchange(
        sizeof(lon_ib_con_t), reinterpret_cast<char*>(&local_con_prop), reinterpret_cast<char*>(&temp_con_prop));
    CHECK(ret >= 0) << "Failed to exchange connection data while filling remote "
                       "properties for ack: "
            + nodename + "/" + res->get_device()->get_name();

    remote_con_prop.addr = ntohll(temp_con_prop.addr);
    remote_con_prop.rkey = ntohl(temp_con_prop.rkey);
    remote_con_prop.qp_num = ntohl(temp_con_prop.qp_num);
    remote_con_prop.lid = ntohs(temp_con_prop.lid);
    remote_con_prop.canary = ntohs(temp_con_prop.canary);

    // save the remote side attributes, we will need it for the post SR
    res->set_remote_ack_properties(remote_con_prop);

    // print properties
    res->print_remote_properties();

    return ret;
}

int IBConnections::connect_tcp(
    const std::string& nodename, const std::string& port, addrinfo* resolved_addr, bool is_SUT)
{
    int sock = -1, lstn = -1;
    char service[6] = {0, 0, 0, 0, 0, 0};

    int ret;

    addrinfo hints
        = {.ai_flags = AI_PASSIVE, .ai_family = AF_INET, .ai_socktype = SOCK_STREAM, .ai_protocol = IPPROTO_TCP};

    // Resolve address
    ret = getaddrinfo(nodename.c_str(), port.c_str(), &hints, &resolved_addr);
    CHECK(!ret) << std::to_string(ret) + ", error: " + std::string(gai_strerror(ret)) + " for " + nodename + ":" + port
            + " failed";

    // getaddrinfo returns one or more addrinfo structures; use the first one only
    sockaddr_in* addr_in = (struct sockaddr_in*) resolved_addr->ai_addr;
    std::string res_addr(inet_ntoa(addr_in->sin_addr));
    gLogVerbose << "Resolved IP address: " << res_addr << std::endl;
    sock = socket(resolved_addr->ai_family, resolved_addr->ai_socktype, resolved_addr->ai_protocol);
    gLogVerbose << res_addr << " -- preparing socket: " << sock << std::endl;
    if (sock >= 0)
    {
        sockaddr_in* sut_addr = reinterpret_cast<sockaddr_in*>(resolved_addr->ai_addr);
        int sut_port = stoi(port);
        sut_addr->sin_family = resolved_addr->ai_family;
        if (is_SUT)
        {
            // SUT (server); waiting for LON to connect
            // bind the socket for listening
            lstn = sock;
            sock = -1;
            sut_addr->sin_addr.s_addr = htonl(INADDR_ANY);
            ret = bind(lstn, reinterpret_cast<sockaddr*>(sut_addr), resolved_addr->ai_addrlen);
            if (ret)
            {
                // failed to bind port
                close(lstn);
                CHECK(false) << "Failed to bind and listen to port " + port;
            }
            gLogInfo << "SUT[" << res_addr << "]:" << ntohs(sut_addr->sin_port) << " bound for TCP connection"
                     << std::endl;
            listen(lstn, 1);
            gLogInfo << "SUT[" << res_addr << "]:" << ntohs(sut_addr->sin_port) << " listening for TCP connection"
                     << std::endl;
            sock = accept(lstn, NULL, 0);
            if (sock < 0)
            {
                // failed to accept incoming connection
                close(lstn);
                CHECK(false) << "Failed to accept on port " + ntohs(sut_addr->sin_port);
            }
            gLogInfo << "SUT[" << res_addr << "]:" << ntohs(sut_addr->sin_port) << " accepted TCP connection"
                     << std::endl;
        }
        else
        {
            // LON (client)
            uint64_t start_time_msec, current_time_msec;
            start_time_msec = current_time_msec = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                                                      .count();
            ret = -1;
            bool timeout = false;
            while ((ret != 0) && !timeout)
            {
                ret = connect(sock, reinterpret_cast<sockaddr*>(sut_addr), resolved_addr->ai_addrlen);
                current_time_msec = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch())
                                        .count();
                timeout = USE_POLL_TIMEOUT && (current_time_msec - start_time_msec) > TCP_POLL_TIMEOUT_MS;
            }

            CHECK(!timeout) << "Failure: Poll TCP timeout";
            gLogInfo << "LON connecting to " << res_addr << ":" << ntohs(sut_addr->sin_port) << ", socket " << sock
                     << std::endl;
            if (ret)
            {
                // failed to connect to SUT
                close(sock);
                CHECK(false) << std::to_string(ret) + ", LON failed to connect to SUT[" + nodename
                        + "]:" + std::to_string(ntohs(sut_addr->sin_port));
            }
            gLogInfo << "LON Connected to " << res_addr << ":" << ntohs(sut_addr->sin_port) << std::endl;
        }
    }

    return sock;
}

int IBConnections::sock_data_exchange(int transfer_size, char* local_data, char* remote_data)
{
    int ret{0};
    std::size_t my_size{0}, offset{0};

    while (offset < transfer_size)
    {
        my_size = write(sock, local_data + offset, transfer_size - offset);
        CHECK(offset >= 0) << "Socket write failed with errno: " + std::string(std::strerror(errno));
        if (my_size == 0)
            break;
        else
            offset += my_size;
    }
    ret += offset;

    offset = 0;
    while (offset < transfer_size)
    {
        my_size = read(sock, remote_data + offset, transfer_size - offset);
        CHECK(offset >= 0) << "Socket read failed with errno: " + std::string(std::strerror(errno));
        if (my_size == 0)
            break;
        else
            offset += my_size;
    }
    ret += offset;

    return ret;
}

} // namespace lonib