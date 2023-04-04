#include "dnnl.hpp"
#include "dnnl_types.h"
#include <cstring>
#include <iostream>

const int Start_Out_C = 64;
const int Start_Out_H = 112;
const int Start_Out_W = 112;
const int Start_W_H = 7;
const int Start_W_W = 7;
const int Start_I_H = 224;
const int Start_I_W = 224;
const int Start_In_C = 3;

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

static dnnl::engine eng_dnn_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
static dnnl::stream dnn_strm_ = dnnl::stream(eng_dnn_);

static dnnl::memory::dims conv_src_tz_stg1_ = {1, Start_In_C, Start_I_H, Start_I_W};
static dnnl::memory::dims conv_weights_tz_stg1_ = {Start_Out_C, Start_In_C, Start_W_H, Start_W_W};
static dnnl::memory::dims conv_bias_tz_stg1_ = {Start_Out_C};
static dnnl::memory::dims conv_dst_tz_stg1_ = {1, Start_Out_C, Start_Out_H, Start_Out_W};
static dnnl::memory::dims conv_strides_stg1_ = {2, 2};
static dnnl::memory::dims conv_padding_stg1_ = {3, 3};
static dnnl::memory::dims maxpool_dst_tz_ = {1, 64, 56, 56};
static dnnl::memory::dims maxpool_kernel_sz_ = {3,3};
static dnnl::memory::dims maxpool_strides_sz_ = {2,2};
static dnnl::memory::dims maxpool_padding_sz_ = {1,1};

static int8_t* conv_weights_ptr_stg1_ = (int8_t*) aligned_alloc(size_t(64), 64*3*7*7*sizeof(int8_t));
static float* conv_bias_ptr_stg1_ = (float*) aligned_alloc(size_t(64), 64*sizeof(float));

static int8_t* fc_weights_ptr_end_ = (int8_t*) aligned_alloc(size_t(64), 2097152*sizeof(int8_t));
static float* fc_bias_ptr_end_ = (float*) aligned_alloc(size_t(64), 1000*sizeof(float));


static dnnl::convolution_forward conv_forward_prim_;
static dnnl::pooling_forward pool_forward_prim_;
static dnnl::memory::desc scratchpad_md_prim_;

static dnnl::inner_product_forward fc_forward_prim_;
static dnnl::pooling_forward avg_pool_forward_prim_;
static dnnl::reorder fc_input_reorder;
static dnnl::memory::desc post_scratchpad_md_prim_, reorder_scratchpad_md_8;

static dnnl::memory conv_weights_memory, conv_bias_memory;

static dnnl::memory fc_weights_memory;
static dnnl::memory fc_bias_memory;


static bool init_onednn = false;
