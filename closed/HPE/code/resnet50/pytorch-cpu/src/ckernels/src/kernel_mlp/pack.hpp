#pragma once
#include "reorder.hpp"
#include <string.h>

namespace mlp_fwd_4k_shape {
inline void pack_bias0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_bias0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_bias1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_bias1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_bias2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_bias2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_relu_out0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2097152);
}

inline void unpack_relu_out0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2097152);
}

inline void pack_relu_out1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_relu_out1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_weight0(uint16_t* out, uint16_t* in) {
    reorder_13x512_AB_BA16a64b2a(out, in);
}

inline void unpack_weight0(uint16_t* out, uint16_t* in) {
    reorder_13x512_BA16a64b2a_AB(out, in);
}

inline void pack_weight1(uint16_t* out, uint16_t* in) {
    reorder_512x256_AB_BA64a64b2a(out, in);
}

inline void unpack_weight1(uint16_t* out, uint16_t* in) {
    reorder_512x256_BA64a64b2a_AB(out, in);
}

inline void pack_weight2(uint16_t* out, uint16_t* in) {
    reorder_256x128_AB_BA64a64b2a(out, in);
}

inline void unpack_weight2(uint16_t* out, uint16_t* in) {
    reorder_256x128_BA64a64b2a_AB(out, in);
}

}

namespace mlp_bwd_4k_shape {
inline void pack_data_input0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 53248);
}

inline void unpack_data_input0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 53248);
}

inline void pack_data_input1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2097152);
}

inline void unpack_data_input1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2097152);
}

inline void pack_data_input2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_data_input2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_gradient(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void unpack_gradient(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void pack_in_relu_output(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void unpack_in_relu_output(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void pack_out_grad_bias0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_out_grad_bias0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_out_grad_bias1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_out_grad_bias1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_out_grad_bias2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_out_grad_bias2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_out_grad_input0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 53248);
}

inline void unpack_out_grad_input0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 53248);
}

inline void pack_out_grad_weight0(uint16_t* out, uint16_t* in) {
    reorder_13x512_AB_BA16a64b2a(out, in);
}

inline void unpack_out_grad_weight0(uint16_t* out, uint16_t* in) {
    reorder_13x512_BA16a64b2a_AB(out, in);
}

inline void pack_out_grad_weight1(uint16_t* out, uint16_t* in) {
    reorder_512x256_AB_BA64a64b2a(out, in);
}

inline void unpack_out_grad_weight1(uint16_t* out, uint16_t* in) {
    reorder_512x256_BA64a64b2a_AB(out, in);
}

inline void pack_out_grad_weight2(uint16_t* out, uint16_t* in) {
    reorder_256x128_AB_BA64a64b2a(out, in);
}

inline void unpack_out_grad_weight2(uint16_t* out, uint16_t* in) {
    reorder_256x128_BA64a64b2a_AB(out, in);
}

inline void pack_weight_input0(uint16_t* out, uint16_t* in) {
    reorder_13x512_AB_BA16a64b2a(out, in);
}

inline void unpack_weight_input0(uint16_t* out, uint16_t* in) {
    reorder_13x512_BA16a64b2a_AB(out, in);
}

inline void pack_weight_input1(uint16_t* out, uint16_t* in) {
    reorder_512x256_AB_BA64a64b2a(out, in);
}

inline void unpack_weight_input1(uint16_t* out, uint16_t* in) {
    reorder_512x256_BA64a64b2a_AB(out, in);
}

inline void pack_weight_input2(uint16_t* out, uint16_t* in) {
    reorder_256x128_AB_BA64a64b2a(out, in);
}

inline void unpack_weight_input2(uint16_t* out, uint16_t* in) {
    reorder_256x128_BA64a64b2a_AB(out, in);
}

}


namespace mlp_fwd_128k_shape {
inline void pack_bias0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_bias0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_bias1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_bias1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_bias2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_bias2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_relu_out0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 67108864);
}

inline void unpack_relu_out0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 67108864);
}

inline void pack_relu_out1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 33554432);
}

inline void unpack_relu_out1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 33554432);
}

inline void pack_weight0(uint16_t* out, uint16_t* in) {
    reorder_13x512_AB_BA16a64b2a(out, in);
}

inline void unpack_weight0(uint16_t* out, uint16_t* in) {
    reorder_13x512_BA16a64b2a_AB(out, in);
}

inline void pack_weight1(uint16_t* out, uint16_t* in) {
    reorder_512x256_AB_BA64a64b2a(out, in);
}

inline void unpack_weight1(uint16_t* out, uint16_t* in) {
    reorder_512x256_BA64a64b2a_AB(out, in);
}

inline void pack_weight2(uint16_t* out, uint16_t* in) {
    reorder_256x128_AB_BA64a64b2a(out, in);
}

inline void unpack_weight2(uint16_t* out, uint16_t* in) {
    reorder_256x128_BA64a64b2a_AB(out, in);
}

}

namespace mlp_bwd_128k_shape {
inline void pack_data_input0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1703936);
}

inline void unpack_data_input0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1703936);
}

inline void pack_data_input1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 67108864);
}

inline void unpack_data_input1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 67108864);
}

inline void pack_data_input2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 33554432);
}

inline void unpack_data_input2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 33554432);
}

inline void pack_gradient(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16777216);
}

inline void unpack_gradient(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16777216);
}

inline void pack_in_relu_output(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16777216);
}

inline void unpack_in_relu_output(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16777216);
}

inline void pack_out_grad_bias0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_out_grad_bias0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_out_grad_bias1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_out_grad_bias1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_out_grad_bias2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_out_grad_bias2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_out_grad_input0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1703936);
}

inline void unpack_out_grad_input0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1703936);
}

inline void pack_out_grad_weight0(uint16_t* out, uint16_t* in) {
    reorder_13x512_AB_BA16a64b2a(out, in);
}

inline void unpack_out_grad_weight0(uint16_t* out, uint16_t* in) {
    reorder_13x512_BA16a64b2a_AB(out, in);
}

inline void pack_out_grad_weight1(uint16_t* out, uint16_t* in) {
    reorder_512x256_AB_BA64a64b2a(out, in);
}

inline void unpack_out_grad_weight1(uint16_t* out, uint16_t* in) {
    reorder_512x256_BA64a64b2a_AB(out, in);
}

inline void pack_out_grad_weight2(uint16_t* out, uint16_t* in) {
    reorder_256x128_AB_BA64a64b2a(out, in);
}

inline void unpack_out_grad_weight2(uint16_t* out, uint16_t* in) {
    reorder_256x128_BA64a64b2a_AB(out, in);
}

inline void pack_weight_input0(uint16_t* out, uint16_t* in) {
    reorder_13x512_AB_BA16a64b2a(out, in);
}

inline void unpack_weight_input0(uint16_t* out, uint16_t* in) {
    reorder_13x512_BA16a64b2a_AB(out, in);
}

inline void pack_weight_input1(uint16_t* out, uint16_t* in) {
    reorder_512x256_AB_BA64a64b2a(out, in);
}

inline void unpack_weight_input1(uint16_t* out, uint16_t* in) {
    reorder_512x256_BA64a64b2a_AB(out, in);
}

inline void pack_weight_input2(uint16_t* out, uint16_t* in) {
    reorder_256x128_AB_BA64a64b2a(out, in);
}

inline void unpack_weight_input2(uint16_t* out, uint16_t* in) {
    reorder_256x128_BA64a64b2a_AB(out, in);
}

}

