/*
 * Copyright (c) 2023, NEUCHIPS.  All rights reserved.
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
#ifndef NEUCHIPS_USER_H
#define NEUCHIPS_USER_H

#include<sys/ioctl.h>
#include <stdio.h>
#include <sys/types.h>

#include <stdbool.h>

enum INPUT_UNIT_IDX {
    ALIGN_PING_IDX = 0,
    EMB_TOP_PING_IDX,
    EMB_BOTTOM_PING_IDX,
    ALIGN_PONG_IDX,
    EMB_TOP_PONG_IDX,
    EMB_BOTTOM_PONG_IDX,
};

enum PCIE_DIRECT {
    READ_EP = 0,
    WRITE_EP = 1,

};

typedef unsigned char    u8;
typedef unsigned short  u16;
typedef unsigned int    u32;
typedef unsigned long   dma_addr_t;

enum IOCTL_NUM {
    IOCTL_SET_BAR_TO_CONTROL = 0,
    IOCTL_DMA_CH_SET,
    IOCTL_GET_DMA_WR_ADDR,
    IOCTL_WRITE_DATA,
    IOCTL_READ_DATA,
    IOCTL_FREE_DMA_CHANNEL,
    IOCTL_DLRM_READ_CH_SET,
    IOCTL_GET_RESULT,
    IOCTL_PREDICT,
    IOCTL_FREE_RD_CHANNEL,
    IOCTL_DMA_BUF_INIT,
    IOCTL_FILL_DMA_DATA,
    IOCTL_DMA_BUF_RELEASE,
    IOCTL_H2D_RENEW_SGL,
    IOCTL_DLRM_WRITE_DATA,
    IOCTL_GET_DMA_BUF_MMAP_ADDR,
    IOCTL_PRINT_CURRENT_INFO,
    IOCTL_H2D_RENEW_SGL_ALIGNER,
    IOCTL_H2D_RENEW_SGL_EMB
};


#define IOC_MAGIC                0x77
#define IOCX_SET_BAR_TO_CONTROL  _IOW(IOC_MAGIC, IOCTL_SET_BAR_TO_CONTROL,struct neuchip_ioctl_arg)

#define IOCX_DMA_CH_SET         _IOWR(IOC_MAGIC, IOCTL_DMA_CH_SET,       struct neuchip_ioctl_arg)
#define IOCX_GET_DMA_WR_ADDR    _IOWR(IOC_MAGIC, IOCTL_GET_DMA_WR_ADDR,  struct neuchip_ioctl_arg)
#define IOCX_WRITE_DATA         _IOW(IOC_MAGIC, IOCTL_WRITE_DATA,       struct neuchip_ioctl_arg)
#define IOCX_READ_DATA          _IOWR(IOC_MAGIC, IOCTL_READ_DATA,        struct neuchip_ioctl_arg)
#define IOCX_FREE_DMA_CHANNEL   _IOW(IOC_MAGIC, IOCTL_FREE_DMA_CHANNEL, struct neuchip_ioctl_arg)
#define IOCX_DLRM_READ_CH_SET   _IOWR(IOC_MAGIC, IOCTL_DLRM_READ_CH_SET,struct neuchip_ioctl_arg)

#define IOCX_PREDICT            _IOW(IOC_MAGIC, IOCTL_PREDICT,          struct neuchip_ioctl_arg)
#define IOCX_DMA_BUF_INIT       _IOW(IOC_MAGIC, IOCTL_DMA_BUF_INIT,     struct ncs_arg_dma_buf_init*)
#define IOCX_FILL_DMA_DATA      _IOW(IOC_MAGIC, IOCTL_FILL_DMA_DATA,    struct ncs_arg_fill_dma_data)
#define IOCX_DMA_BUF_RELEASE    _IO(IOC_MAGIC, IOCTL_DMA_BUF_RELEASE )
#define IOCX_H2D_RENEW_SGL      _IOW(IOC_MAGIC, IOCTL_H2D_RENEW_SGL,    struct ncs_arg_h2d_renew_sgl)
#define IOCX_GET_RESULT         _IOWR(IOC_MAGIC, IOCTL_GET_RESULT,       struct neuchip_ioctl_arg)
#define IOCX_GET_DMA_BUF_MMAP_ADDR _IOWR(IOC_MAGIC, IOCTL_GET_DMA_BUF_MMAP_ADDR,  struct dma_mmap_param)

#define IOCX_PRINT_CURRENT_INFO     _IOW(IOC_MAGIC, IOCTL_PRINT_CURRENT_INFO, struct neuchip_ioctl_arg)
#define IOCX_H2D_RENEW_SGL_ALIGNER  _IOW(IOC_MAGIC, IOCTL_H2D_RENEW_SGL_ALIGNER,  struct ncs_arg_h2d_renew_sgl)
#define IOCX_H2D_RENEW_SGL_EMB      _IOW(IOC_MAGIC, IOCTL_H2D_RENEW_SGL_EMB,    struct ncs_arg_h2d_renew_sgl)

struct lookup_entry {
    u32         index;
    dma_addr_t  phy_addr;
    u32         len;
    void*       vaddr;
};

struct ncs_arg_dma_buf_init {
    u32                  numa_node;
    u32                  nents;
    struct lookup_entry* tbl;
    u32*                 partition_sizes;
};

struct ncs_arg_fill_dma_data{
    u32         input_index;
    u32         sample_index;
    u32         len;
    u8*         user_host_buffer;
};

struct ncs_arg_h2d_renew_sgl {
    u32 chan_idx;
    u32 nents;
    u32* h2d_indices;
};

struct ncs_arg_write{
    u32        chan_no;
};

struct dma_mmap_param{
    int dma_mmap_en;
    int dma_chan;
    bool rd_wr;
    unsigned long dma_buf_idx;
    unsigned long mmap_size;
};

struct neuchip_ioctl_arg {
    int bar_no;
    int dma_chan;
    unsigned long dma_buf_len;
    unsigned long dma_buf_idx;
    unsigned int  offset;
    unsigned long wr_buf_ofs;
    unsigned long rd_buf_ofs;
    bool rd_wr;

    //  dlrm setting
    unsigned char* rd_buf;
    unsigned char  ping_pong_sel;
    bool           final_send;

};

#endif /* NEUCHIPS_USER_H */

