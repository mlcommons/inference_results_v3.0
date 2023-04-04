#pragma once
#include "reorder.hpp"
#include <string.h>

namespace backbone_256_shape {
inline void pack_backbone_input(uint16_t* out, uint16_t* in) {
    reorder_256x64x56x56_ABCD_ACDB(out, in);
}

inline void unpack_backbone_input(uint16_t* out, uint16_t* in) {
    reorder_256x64x56x56_ACDB_ABCD(out, in);
}

inline void pack_backbone_output(uint16_t* out, uint16_t* in) {
    reorder_256x2048x7x7_ABCD_ACDB(out, in);
}

inline void unpack_backbone_output(uint16_t* out, uint16_t* in) {
    reorder_256x2048x7x7_ACDB_ABCD(out, in);
}

inline void pack_res2a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 4096);
}

inline void unpack_res2a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 4096);
}

inline void pack_res2a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void unpack_res2a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void pack_res2a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void unpack_res2b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void pack_res2b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void unpack_res2c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void pack_res2c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res3a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 32768);
}

inline void unpack_res3a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 32768);
}

inline void pack_res3a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void unpack_res3a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void pack_res3b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res4a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void unpack_res4a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void pack_res4a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void unpack_res4a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void pack_res4b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4e_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4e_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4e_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4e_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4e_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4e_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4e_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4e_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4e_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4e_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4e_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4e_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4f_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4f_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4f_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4f_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4f_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4f_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4f_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4f_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4f_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4f_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4f_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4f_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res5a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void unpack_res5a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void pack_res5a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void unpack_res5a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void pack_res5a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2097152);
}

inline void unpack_res5a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2097152);
}

inline void pack_res5b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void unpack_res5b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void pack_res5b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void unpack_res5c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void pack_res5c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

}


namespace backbone_8_shape {
inline void pack_backbone_input(uint16_t* out, uint16_t* in) {
    reorder_8x64x56x56_ABCD_ACDB(out, in);
}

inline void unpack_backbone_input(uint16_t* out, uint16_t* in) {
    reorder_8x64x56x56_ACDB_ABCD(out, in);
}

inline void pack_backbone_output(uint16_t* out, uint16_t* in) {
    reorder_8x2048x7x7_ABCD_ACDB(out, in);
}

inline void unpack_backbone_output(uint16_t* out, uint16_t* in) {
    reorder_8x2048x7x7_ACDB_ABCD(out, in);
}

inline void pack_res2a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 4096);
}

inline void unpack_res2a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 4096);
}

inline void pack_res2a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void unpack_res2a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void pack_res2a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void unpack_res2b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void pack_res2b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void unpack_res2c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void pack_res2c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res3a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 32768);
}

inline void unpack_res3a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 32768);
}

inline void pack_res3a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void unpack_res3a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void pack_res3b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res4a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void unpack_res4a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void pack_res4a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void unpack_res4a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void pack_res4b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4e_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4e_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4e_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4e_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4e_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4e_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4e_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4e_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4e_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4e_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4e_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4e_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4f_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4f_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4f_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4f_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4f_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4f_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4f_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4f_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4f_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4f_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4f_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4f_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res5a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void unpack_res5a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void pack_res5a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void unpack_res5a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void pack_res5a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2097152);
}

inline void unpack_res5a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2097152);
}

inline void pack_res5b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void unpack_res5b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void pack_res5b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void unpack_res5c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void pack_res5c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

}


namespace backbone_4_shape {
inline void pack_backbone_input(uint16_t* out, uint16_t* in) {
    reorder_4x64x56x56_ABCD_ACDB(out, in);
}

inline void unpack_backbone_input(uint16_t* out, uint16_t* in) {
    reorder_4x64x56x56_ACDB_ABCD(out, in);
}

inline void pack_backbone_output(uint16_t* out, uint16_t* in) {
    reorder_4x2048x7x7_ABCD_ACDB(out, in);
}

inline void unpack_backbone_output(uint16_t* out, uint16_t* in) {
    reorder_4x2048x7x7_ACDB_ABCD(out, in);
}

inline void pack_res2a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 4096);
}

inline void unpack_res2a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 4096);
}

inline void pack_res2a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void unpack_res2a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void pack_res2a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void unpack_res2b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void pack_res2b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void unpack_res2c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 64);
}

inline void pack_res2c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res2c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res2c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res2c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void unpack_res2c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 36864);
}

inline void pack_res2c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void unpack_res2c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 16384);
}

inline void pack_res3a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 32768);
}

inline void unpack_res3a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 32768);
}

inline void pack_res3a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void unpack_res3a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void pack_res3b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void unpack_res3d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 128);
}

inline void pack_res3d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res3d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res3d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res3d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void unpack_res3d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 147456);
}

inline void pack_res3d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void unpack_res3d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 65536);
}

inline void pack_res4a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void unpack_res4a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 131072);
}

inline void pack_res4a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void unpack_res4a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void pack_res4b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4d_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4d_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4d_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4d_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4d_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4d_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4e_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4e_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4e_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4e_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4e_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4e_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4e_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4e_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4e_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4e_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4e_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4e_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4f_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4f_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4f_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void unpack_res4f_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 256);
}

inline void pack_res4f_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void unpack_res4f_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1024);
}

inline void pack_res4f_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4f_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res4f_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void unpack_res4f_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 589824);
}

inline void pack_res4f_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void unpack_res4f_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 262144);
}

inline void pack_res5a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5a_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5a_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5a_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5a_bias_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void unpack_res5a_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 524288);
}

inline void pack_res5a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void unpack_res5a_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void pack_res5a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5a_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2097152);
}

inline void unpack_res5a_weight_b(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2097152);
}

inline void pack_res5b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5b_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5b_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5b_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5b_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void unpack_res5b_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void pack_res5b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5b_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5c_bias_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void unpack_res5c_bias_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 512);
}

inline void pack_res5c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void unpack_res5c_bias_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2048);
}

inline void pack_res5c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5c_weight_0(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void pack_res5c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void unpack_res5c_weight_1(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 2359296);
}

inline void pack_res5c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

inline void unpack_res5c_weight_2(uint16_t* out, uint16_t* in) {
    memcpy(out, in, sizeof(uint16_t) * 1048576);
}

}
