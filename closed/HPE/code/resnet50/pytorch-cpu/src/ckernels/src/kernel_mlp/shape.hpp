#pragma once
#include <array>

namespace mlp_fwd_4k_shape {
inline const std::array<int, 1>& bias0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& bias1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& bias2() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 2>& input() {
    static const std::array<int, 2> shape = {4096,13};
    return shape;
}

inline const std::array<int, 2>& relu_out0() {
    static const std::array<int, 2> shape = {4096,512};
    return shape;
}

inline const std::array<int, 2>& relu_out1() {
    static const std::array<int, 2> shape = {4096,256};
    return shape;
}

inline const std::array<int, 2>& relu_out2() {
    static const std::array<int, 2> shape = {4096,128};
    return shape;
}

inline const std::array<int, 5>& weight0() {
    static const std::array<int, 5> shape = {8,1,8,64,2};
    return shape;
}

inline const std::array<int, 5>& weight1() {
    static const std::array<int, 5> shape = {4,8,32,64,2};
    return shape;
}

inline const std::array<int, 5>& weight2() {
    static const std::array<int, 5> shape = {2,4,32,64,2};
    return shape;
}

}

namespace mlp_bwd_4k_shape {
inline const std::array<int, 2>& data_input0() {
    static const std::array<int, 2> shape = {4096,13};
    return shape;
}

inline const std::array<int, 2>& data_input1() {
    static const std::array<int, 2> shape = {4096,512};
    return shape;
}

inline const std::array<int, 2>& data_input2() {
    static const std::array<int, 2> shape = {4096,256};
    return shape;
}

inline const std::array<int, 2>& gradient() {
    static const std::array<int, 2> shape = {4096,128};
    return shape;
}

inline const std::array<int, 2>& in_relu_output() {
    static const std::array<int, 2> shape = {4096,128};
    return shape;
}

inline const std::array<int, 1>& out_grad_bias0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& out_grad_bias1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& out_grad_bias2() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 2>& out_grad_input0() {
    static const std::array<int, 2> shape = {4096,13};
    return shape;
}

inline const std::array<int, 5>& out_grad_weight0() {
    static const std::array<int, 5> shape = {8,1,8,64,2};
    return shape;
}

inline const std::array<int, 5>& out_grad_weight1() {
    static const std::array<int, 5> shape = {4,8,32,64,2};
    return shape;
}

inline const std::array<int, 5>& out_grad_weight2() {
    static const std::array<int, 5> shape = {2,4,32,64,2};
    return shape;
}

inline const std::array<int, 5>& weight_input0() {
    static const std::array<int, 5> shape = {8,1,8,64,2};
    return shape;
}

inline const std::array<int, 5>& weight_input1() {
    static const std::array<int, 5> shape = {4,8,32,64,2};
    return shape;
}

inline const std::array<int, 5>& weight_input2() {
    static const std::array<int, 5> shape = {2,4,32,64,2};
    return shape;
}

}


namespace mlp_fwd_128k_shape {
inline const std::array<int, 1>& bias0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& bias1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& bias2() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 2>& input() {
    static const std::array<int, 2> shape = {131072,13};
    return shape;
}

inline const std::array<int, 2>& relu_out0() {
    static const std::array<int, 2> shape = {131072,512};
    return shape;
}

inline const std::array<int, 2>& relu_out1() {
    static const std::array<int, 2> shape = {131072,256};
    return shape;
}

inline const std::array<int, 2>& relu_out2() {
    static const std::array<int, 2> shape = {131072,128};
    return shape;
}

inline const std::array<int, 5>& weight0() {
    static const std::array<int, 5> shape = {8,1,8,64,2};
    return shape;
}

inline const std::array<int, 5>& weight1() {
    static const std::array<int, 5> shape = {4,8,32,64,2};
    return shape;
}

inline const std::array<int, 5>& weight2() {
    static const std::array<int, 5> shape = {2,4,32,64,2};
    return shape;
}

}

namespace mlp_bwd_128k_shape {
inline const std::array<int, 2>& data_input0() {
    static const std::array<int, 2> shape = {131072,13};
    return shape;
}

inline const std::array<int, 2>& data_input1() {
    static const std::array<int, 2> shape = {131072,512};
    return shape;
}

inline const std::array<int, 2>& data_input2() {
    static const std::array<int, 2> shape = {131072,256};
    return shape;
}

inline const std::array<int, 2>& gradient() {
    static const std::array<int, 2> shape = {131072,128};
    return shape;
}

inline const std::array<int, 2>& in_relu_output() {
    static const std::array<int, 2> shape = {131072,128};
    return shape;
}

inline const std::array<int, 1>& out_grad_bias0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& out_grad_bias1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& out_grad_bias2() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 2>& out_grad_input0() {
    static const std::array<int, 2> shape = {131072,13};
    return shape;
}

inline const std::array<int, 5>& out_grad_weight0() {
    static const std::array<int, 5> shape = {8,1,8,64,2};
    return shape;
}

inline const std::array<int, 5>& out_grad_weight1() {
    static const std::array<int, 5> shape = {4,8,32,64,2};
    return shape;
}

inline const std::array<int, 5>& out_grad_weight2() {
    static const std::array<int, 5> shape = {2,4,32,64,2};
    return shape;
}

inline const std::array<int, 5>& weight_input0() {
    static const std::array<int, 5> shape = {8,1,8,64,2};
    return shape;
}

inline const std::array<int, 5>& weight_input1() {
    static const std::array<int, 5> shape = {4,8,32,64,2};
    return shape;
}

inline const std::array<int, 5>& weight_input2() {
    static const std::array<int, 5> shape = {2,4,32,64,2};
    return shape;
}

}

