# MLPerf Inference v3.0 - Alibaba - Calibration Details

## Alibaba Cloud Server E-Series SinianML Platform - Calibration

### SinianML Quantization Aware Training (QAT)
SinianML QAT employs per-channel symmetric quantization for weight tensors and per-tensor asymmetric quantization for activation tensors.

### Activation
`Per-tensor asymmetric` quantization is used. Collecting activation the min/max value with validation datasets listed in MLPerf. Next, train quantization parameters through SinianML QAT. Activation tensors were quantized to `int8`.

### Weights
`Per-channel symmetric` quantization is used. First, find the maximum absolute value of each output channel of the weight tensors. Second, train the quantized model with SinianML QAT method in training datasets listed in MLPerf. Weight tensors were quantized to `int8` and bias tensors were quantized to `int32`.

## Additional Details

We use Pytorch>=1.9.0 and SinianML QAT for quantization.

## Reference

https://pytorch.org/docs/stable/quantization.html
