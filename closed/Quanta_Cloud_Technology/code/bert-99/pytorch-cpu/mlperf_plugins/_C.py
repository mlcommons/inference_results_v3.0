import os
import sys
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))

if sys.platform == 'darwin':
    torch.ops.load_library(script_dir + "/build/libmlperf_plugins.dylib")
elif sys.platform == 'linux':
    torch.ops.load_library(script_dir + "/build/libmlperf_plugins.so")

linear = torch.ops.intel_mlperf.linear
linear_gelu = torch.ops.intel_mlperf.linear_gelu
baddbmm_out_ = torch.ops.intel_mlperf.baddbmm_out_
prepack_linear_weight = torch.ops.intel_mlperf.prepack_linear_weight
matmul_out_ = torch.ops.intel_mlperf.matmul_out_
reorder_test = torch.ops.intel_mlperf.reorder_test
i_softmax = torch.ops.intel_mlperf.i_softmax
i_softmax_u = torch.ops.intel_mlperf.i_softmax_u
i_gelu = torch.ops.intel_mlperf.i_gelu
i_identity = torch.ops.intel_mlperf.i_identity
i_identity_cin = torch.ops.intel_mlperf.i_identity_cin
i_identity_ = torch.ops.intel_mlperf.i_identity_
i_layernorm = torch.ops.intel_mlperf.i_layernorm
i_residual_layernorm = torch.ops.intel_mlperf.i_residual_layernorm
i_residual_layernorm_ = torch.ops.intel_mlperf.i_residual_layernorm_
i_residual_layernorm_cin_ = torch.ops.intel_mlperf.i_residual_layernorm_cin_
amx_mha = torch.ops.intel_mlperf.amx_mha
amx_mha_concat = torch.ops.intel_mlperf.amx_mha_concat
amx_linear = torch.ops.intel_mlperf.amx_linear
