#pragma once
#include <stdint.h>


/**
 * mlp_training_forward_4k
 * @param relu_out0 output tensor, [bf16 [4096, 512] @ AB]
 * @param relu_out1 output tensor, [bf16 [4096, 256] @ AB]
 * @param relu_out2 output tensor, [bf16 [4096, 128] @ AB]
 * @param input input tensor, [bf16 [4096, 13] @ AB]
 * @param weight0 input tensor, [bf16 [13, 512] @ BA16a64b2a], real dims is [8, 1, 8, 64, 2]
 * @param bias0 input tensor, [bf16 [512] @ A]
 * @param weight1 input tensor, [bf16 [512, 256] @ BA64a64b2a], real dims is [4, 8, 32, 64, 2]
 * @param bias1 input tensor, [bf16 [256] @ A]
 * @param weight2 input tensor, [bf16 [256, 128] @ BA64a64b2a], real dims is [2, 4, 32, 64, 2]
 * @param bias2 input tensor, [bf16 [128] @ A]
**/
extern "C" void mlp_training_forward_4k(uint16_t* relu_out0, uint16_t* relu_out1, uint16_t* relu_out2, uint16_t* input, uint16_t* weight0, uint16_t* bias0, uint16_t* weight1, uint16_t* bias1, uint16_t* weight2, uint16_t* bias2);

extern "C" void sc_init_mlp_training_forward_4k();

/**
 * mlp_training_backward_4k
 * @param out_grad_bias2 output tensor, [bf16 [128] @ A]
 * @param out_grad_weight2 output tensor, [bf16 [256, 128] @ BA64a64b2a], real dims is [2, 4, 32, 64, 2]
 * @param out_grad_bias1 output tensor, [bf16 [256] @ A]
 * @param out_grad_weight1 output tensor, [bf16 [512, 256] @ BA64a64b2a], real dims is [4, 8, 32, 64, 2]
 * @param out_grad_bias0 output tensor, [bf16 [512] @ A]
 * @param out_grad_weight0 output tensor, [bf16 [13, 512] @ BA16a64b2a], real dims is [8, 1, 8, 64, 2]
 * @param out_grad_input0 output tensor, [bf16 [4096, 13] @ AB]
 * @param gradient input tensor, [bf16 [4096, 128] @ AB]
 * @param in_relu_output input tensor, [bf16 [4096, 128] @ AB]
 * @param data_input2 input tensor, [bf16 [4096, 256] @ AB]
 * @param weight_input2 input tensor, [bf16 [256, 128] @ BA64a64b2a], real dims is [2, 4, 32, 64, 2]
 * @param data_input1 input tensor, [bf16 [4096, 512] @ AB]
 * @param weight_input1 input tensor, [bf16 [512, 256] @ BA64a64b2a], real dims is [4, 8, 32, 64, 2]
 * @param data_input0 input tensor, [bf16 [4096, 13] @ AB]
 * @param weight_input0 input tensor, [bf16 [13, 512] @ BA16a64b2a], real dims is [8, 1, 8, 64, 2]
**/
extern "C" void mlp_training_backward_4k(uint16_t* out_grad_bias2, uint16_t* out_grad_weight2, uint16_t* out_grad_bias1, uint16_t* out_grad_weight1, uint16_t* out_grad_bias0, uint16_t* out_grad_weight0, uint16_t* out_grad_input0, uint16_t* gradient, uint16_t* in_relu_output, uint16_t* data_input2, uint16_t* weight_input2, uint16_t* data_input1, uint16_t* weight_input1, uint16_t* data_input0, uint16_t* weight_input0);

extern "C" void sc_init_mlp_training_backward_4k();

/**
 * mlp_training_forward_128k
 * @param relu_out0 output tensor, [bf16 [131072, 512] @ AB]
 * @param relu_out1 output tensor, [bf16 [131072, 256] @ AB]
 * @param relu_out2 output tensor, [bf16 [131072, 128] @ AB]
 * @param input input tensor, [bf16 [131072, 13] @ AB]
 * @param weight0 input tensor, [bf16 [13, 512] @ BA16a64b2a], real dims is [8, 1, 8, 64, 2]
 * @param bias0 input tensor, [bf16 [512] @ A]
 * @param weight1 input tensor, [bf16 [512, 256] @ BA64a64b2a], real dims is [4, 8, 32, 64, 2]
 * @param bias1 input tensor, [bf16 [256] @ A]
 * @param weight2 input tensor, [bf16 [256, 128] @ BA64a64b2a], real dims is [2, 4, 32, 64, 2]
 * @param bias2 input tensor, [bf16 [128] @ A]
**/
extern "C" void mlp_training_forward_128k(uint16_t* relu_out0, uint16_t* relu_out1, uint16_t* relu_out2, uint16_t* input, uint16_t* weight0, uint16_t* bias0, uint16_t* weight1, uint16_t* bias1, uint16_t* weight2, uint16_t* bias2);

extern "C" void sc_init_mlp_training_forward_128k();

/**
 * mlp_training_backward_128k
 * @param out_grad_bias2 output tensor, [bf16 [128] @ A]
 * @param out_grad_weight2 output tensor, [bf16 [256, 128] @ BA64a64b2a], real dims is [2, 4, 32, 64, 2]
 * @param out_grad_bias1 output tensor, [bf16 [256] @ A]
 * @param out_grad_weight1 output tensor, [bf16 [512, 256] @ BA64a64b2a], real dims is [4, 8, 32, 64, 2]
 * @param out_grad_bias0 output tensor, [bf16 [512] @ A]
 * @param out_grad_weight0 output tensor, [bf16 [13, 512] @ BA16a64b2a], real dims is [8, 1, 8, 64, 2]
 * @param out_grad_input0 output tensor, [bf16 [131072, 13] @ AB]
 * @param gradient input tensor, [bf16 [131072, 128] @ AB]
 * @param in_relu_output input tensor, [bf16 [131072, 128] @ AB]
 * @param data_input2 input tensor, [bf16 [131072, 256] @ AB]
 * @param weight_input2 input tensor, [bf16 [256, 128] @ BA64a64b2a], real dims is [2, 4, 32, 64, 2]
 * @param data_input1 input tensor, [bf16 [131072, 512] @ AB]
 * @param weight_input1 input tensor, [bf16 [512, 256] @ BA64a64b2a], real dims is [4, 8, 32, 64, 2]
 * @param data_input0 input tensor, [bf16 [131072, 13] @ AB]
 * @param weight_input0 input tensor, [bf16 [13, 512] @ BA16a64b2a], real dims is [8, 1, 8, 64, 2]
**/
extern "C" void mlp_training_backward_128k(uint16_t* out_grad_bias2, uint16_t* out_grad_weight2, uint16_t* out_grad_bias1, uint16_t* out_grad_weight1, uint16_t* out_grad_bias0, uint16_t* out_grad_weight0, uint16_t* out_grad_input0, uint16_t* gradient, uint16_t* in_relu_output, uint16_t* data_input2, uint16_t* weight_input2, uint16_t* data_input1, uint16_t* weight_input1, uint16_t* data_input0, uint16_t* weight_input0);

extern "C" void sc_init_mlp_training_backward_128k();
