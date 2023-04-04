#pragma once
#include <array>

namespace backbone_256_shape {
inline const std::array<int, 4>& backbone_input() {
    static const std::array<int, 4> shape = {256,56,56,64};
    return shape;
}

inline const std::array<int, 4>& backbone_output() {
    static const std::array<int, 4> shape = {256,7,7,2048};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_0() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_1() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_2() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_b() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_0() {
    static const std::array<int, 4> shape = {64,64,1,1};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_1() {
    static const std::array<int, 4> shape = {64,64,3,3};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_2() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_b() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 1>& res2b_bias_0() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2b_bias_1() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2b_bias_2() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 4>& res2b_weight_0() {
    static const std::array<int, 4> shape = {64,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res2b_weight_1() {
    static const std::array<int, 4> shape = {64,64,3,3};
    return shape;
}

inline const std::array<int, 4>& res2b_weight_2() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 1>& res2c_bias_0() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2c_bias_1() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2c_bias_2() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 4>& res2c_weight_0() {
    static const std::array<int, 4> shape = {64,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res2c_weight_1() {
    static const std::array<int, 4> shape = {64,64,3,3};
    return shape;
}

inline const std::array<int, 4>& res2c_weight_2() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_b() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_0() {
    static const std::array<int, 4> shape = {128,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_b() {
    static const std::array<int, 4> shape = {512,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res3b_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3b_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3b_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3b_weight_0() {
    static const std::array<int, 4> shape = {128,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res3b_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3b_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 1>& res3c_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3c_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3c_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3c_weight_0() {
    static const std::array<int, 4> shape = {128,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res3c_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3c_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 1>& res3d_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3d_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3d_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3d_weight_0() {
    static const std::array<int, 4> shape = {128,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res3d_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3d_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_b() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_0() {
    static const std::array<int, 4> shape = {256,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_b() {
    static const std::array<int, 4> shape = {1024,512,1,1};
    return shape;
}

inline const std::array<int, 1>& res4b_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4b_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4b_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4b_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4b_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4b_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4c_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4c_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4c_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4c_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4c_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4c_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4d_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4d_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4d_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4d_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4d_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4d_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4e_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4e_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4e_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4e_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4e_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4e_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4f_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4f_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4f_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4f_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4f_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4f_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_1() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_2() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_b() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_0() {
    static const std::array<int, 4> shape = {512,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_1() {
    static const std::array<int, 4> shape = {512,512,3,3};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_2() {
    static const std::array<int, 4> shape = {2048,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_b() {
    static const std::array<int, 4> shape = {2048,1024,1,1};
    return shape;
}

inline const std::array<int, 1>& res5b_bias_0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5b_bias_1() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5b_bias_2() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 4>& res5b_weight_0() {
    static const std::array<int, 4> shape = {512,2048,1,1};
    return shape;
}

inline const std::array<int, 4>& res5b_weight_1() {
    static const std::array<int, 4> shape = {512,512,3,3};
    return shape;
}

inline const std::array<int, 4>& res5b_weight_2() {
    static const std::array<int, 4> shape = {2048,512,1,1};
    return shape;
}

inline const std::array<int, 1>& res5c_bias_0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5c_bias_1() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5c_bias_2() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 4>& res5c_weight_0() {
    static const std::array<int, 4> shape = {512,2048,1,1};
    return shape;
}

inline const std::array<int, 4>& res5c_weight_1() {
    static const std::array<int, 4> shape = {512,512,3,3};
    return shape;
}

inline const std::array<int, 4>& res5c_weight_2() {
    static const std::array<int, 4> shape = {2048,512,1,1};
    return shape;
}

}


namespace backbone_8_shape {
inline const std::array<int, 4>& backbone_input() {
    static const std::array<int, 4> shape = {8,56,56,64};
    return shape;
}

inline const std::array<int, 4>& backbone_output() {
    static const std::array<int, 4> shape = {8,7,7,2048};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_0() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_1() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_2() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_b() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_0() {
    static const std::array<int, 4> shape = {64,64,1,1};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_1() {
    static const std::array<int, 4> shape = {64,64,3,3};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_2() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_b() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 1>& res2b_bias_0() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2b_bias_1() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2b_bias_2() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 4>& res2b_weight_0() {
    static const std::array<int, 4> shape = {64,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res2b_weight_1() {
    static const std::array<int, 4> shape = {64,64,3,3};
    return shape;
}

inline const std::array<int, 4>& res2b_weight_2() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 1>& res2c_bias_0() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2c_bias_1() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2c_bias_2() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 4>& res2c_weight_0() {
    static const std::array<int, 4> shape = {64,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res2c_weight_1() {
    static const std::array<int, 4> shape = {64,64,3,3};
    return shape;
}

inline const std::array<int, 4>& res2c_weight_2() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_b() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_0() {
    static const std::array<int, 4> shape = {128,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_b() {
    static const std::array<int, 4> shape = {512,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res3b_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3b_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3b_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3b_weight_0() {
    static const std::array<int, 4> shape = {128,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res3b_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3b_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 1>& res3c_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3c_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3c_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3c_weight_0() {
    static const std::array<int, 4> shape = {128,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res3c_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3c_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 1>& res3d_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3d_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3d_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3d_weight_0() {
    static const std::array<int, 4> shape = {128,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res3d_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3d_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_b() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_0() {
    static const std::array<int, 4> shape = {256,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_b() {
    static const std::array<int, 4> shape = {1024,512,1,1};
    return shape;
}

inline const std::array<int, 1>& res4b_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4b_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4b_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4b_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4b_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4b_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4c_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4c_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4c_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4c_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4c_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4c_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4d_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4d_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4d_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4d_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4d_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4d_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4e_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4e_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4e_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4e_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4e_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4e_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4f_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4f_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4f_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4f_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4f_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4f_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_1() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_2() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_b() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_0() {
    static const std::array<int, 4> shape = {512,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_1() {
    static const std::array<int, 4> shape = {512,512,3,3};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_2() {
    static const std::array<int, 4> shape = {2048,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_b() {
    static const std::array<int, 4> shape = {2048,1024,1,1};
    return shape;
}

inline const std::array<int, 1>& res5b_bias_0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5b_bias_1() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5b_bias_2() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 4>& res5b_weight_0() {
    static const std::array<int, 4> shape = {512,2048,1,1};
    return shape;
}

inline const std::array<int, 4>& res5b_weight_1() {
    static const std::array<int, 4> shape = {512,512,3,3};
    return shape;
}

inline const std::array<int, 4>& res5b_weight_2() {
    static const std::array<int, 4> shape = {2048,512,1,1};
    return shape;
}

inline const std::array<int, 1>& res5c_bias_0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5c_bias_1() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5c_bias_2() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 4>& res5c_weight_0() {
    static const std::array<int, 4> shape = {512,2048,1,1};
    return shape;
}

inline const std::array<int, 4>& res5c_weight_1() {
    static const std::array<int, 4> shape = {512,512,3,3};
    return shape;
}

inline const std::array<int, 4>& res5c_weight_2() {
    static const std::array<int, 4> shape = {2048,512,1,1};
    return shape;
}

}


namespace backbone_4_shape {
inline const std::array<int, 4>& backbone_input() {
    static const std::array<int, 4> shape = {4,56,56,64};
    return shape;
}

inline const std::array<int, 4>& backbone_output() {
    static const std::array<int, 4> shape = {4,7,7,2048};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_0() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_1() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_2() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res2a_bias_b() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_0() {
    static const std::array<int, 4> shape = {64,64,1,1};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_1() {
    static const std::array<int, 4> shape = {64,64,3,3};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_2() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 4>& res2a_weight_b() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 1>& res2b_bias_0() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2b_bias_1() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2b_bias_2() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 4>& res2b_weight_0() {
    static const std::array<int, 4> shape = {64,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res2b_weight_1() {
    static const std::array<int, 4> shape = {64,64,3,3};
    return shape;
}

inline const std::array<int, 4>& res2b_weight_2() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 1>& res2c_bias_0() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2c_bias_1() {
    static const std::array<int, 1> shape = {64};
    return shape;
}

inline const std::array<int, 1>& res2c_bias_2() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 4>& res2c_weight_0() {
    static const std::array<int, 4> shape = {64,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res2c_weight_1() {
    static const std::array<int, 4> shape = {64,64,3,3};
    return shape;
}

inline const std::array<int, 4>& res2c_weight_2() {
    static const std::array<int, 4> shape = {256,64,1,1};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res3a_bias_b() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_0() {
    static const std::array<int, 4> shape = {128,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 4>& res3a_weight_b() {
    static const std::array<int, 4> shape = {512,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res3b_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3b_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3b_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3b_weight_0() {
    static const std::array<int, 4> shape = {128,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res3b_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3b_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 1>& res3c_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3c_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3c_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3c_weight_0() {
    static const std::array<int, 4> shape = {128,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res3c_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3c_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 1>& res3d_bias_0() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3d_bias_1() {
    static const std::array<int, 1> shape = {128};
    return shape;
}

inline const std::array<int, 1>& res3d_bias_2() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 4>& res3d_weight_0() {
    static const std::array<int, 4> shape = {128,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res3d_weight_1() {
    static const std::array<int, 4> shape = {128,128,3,3};
    return shape;
}

inline const std::array<int, 4>& res3d_weight_2() {
    static const std::array<int, 4> shape = {512,128,1,1};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 1>& res4a_bias_b() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_0() {
    static const std::array<int, 4> shape = {256,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 4>& res4a_weight_b() {
    static const std::array<int, 4> shape = {1024,512,1,1};
    return shape;
}

inline const std::array<int, 1>& res4b_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4b_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4b_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4b_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4b_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4b_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4c_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4c_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4c_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4c_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4c_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4c_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4d_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4d_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4d_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4d_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4d_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4d_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4e_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4e_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4e_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4e_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4e_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4e_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res4f_bias_0() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4f_bias_1() {
    static const std::array<int, 1> shape = {256};
    return shape;
}

inline const std::array<int, 1>& res4f_bias_2() {
    static const std::array<int, 1> shape = {1024};
    return shape;
}

inline const std::array<int, 4>& res4f_weight_0() {
    static const std::array<int, 4> shape = {256,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res4f_weight_1() {
    static const std::array<int, 4> shape = {256,256,3,3};
    return shape;
}

inline const std::array<int, 4>& res4f_weight_2() {
    static const std::array<int, 4> shape = {1024,256,1,1};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_1() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_2() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 1>& res5a_bias_b() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_0() {
    static const std::array<int, 4> shape = {512,1024,1,1};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_1() {
    static const std::array<int, 4> shape = {512,512,3,3};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_2() {
    static const std::array<int, 4> shape = {2048,512,1,1};
    return shape;
}

inline const std::array<int, 4>& res5a_weight_b() {
    static const std::array<int, 4> shape = {2048,1024,1,1};
    return shape;
}

inline const std::array<int, 1>& res5b_bias_0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5b_bias_1() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5b_bias_2() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 4>& res5b_weight_0() {
    static const std::array<int, 4> shape = {512,2048,1,1};
    return shape;
}

inline const std::array<int, 4>& res5b_weight_1() {
    static const std::array<int, 4> shape = {512,512,3,3};
    return shape;
}

inline const std::array<int, 4>& res5b_weight_2() {
    static const std::array<int, 4> shape = {2048,512,1,1};
    return shape;
}

inline const std::array<int, 1>& res5c_bias_0() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5c_bias_1() {
    static const std::array<int, 1> shape = {512};
    return shape;
}

inline const std::array<int, 1>& res5c_bias_2() {
    static const std::array<int, 1> shape = {2048};
    return shape;
}

inline const std::array<int, 4>& res5c_weight_0() {
    static const std::array<int, 4> shape = {512,2048,1,1};
    return shape;
}

inline const std::array<int, 4>& res5c_weight_1() {
    static const std::array<int, 4> shape = {512,512,3,3};
    return shape;
}

inline const std::array<int, 4>& res5c_weight_2() {
    static const std::array<int, 4> shape = {2048,512,1,1};
    return shape;
}

}
