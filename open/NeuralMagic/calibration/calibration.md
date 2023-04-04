# Neural Magic's DeepSparse MLPerf Submission

## ResNet Quantization

- Quantization is carried out using a quantization-aware training (QAT) approach implemented via PyTorch's (De)QuantStubs. MovingAverageMinMax observers are used to estimate quantization ranges during QAT.
- Convolutional layers and add operations are quantized to 8-bits. Other operations (including the matrix-multiply in the classification head) are kept in floating point.
- Batch normalization layers are fused with convolutional layers during QAT such that the fused weights are quantized.
- Activations are quantized with per-tensor asymmetric quantization.
- Weights are quantized with per-tensor symmetric quantization.

## BERT Quantization

- Quantization is carried out using a quantization-aware training (QAT) approach implemented via PyTorch’s (De)QuantStubs. MovingAverageMinMax observers are used to estimate quantization ranges during QAT.
- Gemm and embedding layers are quantized to 8-bits. Other operations are kept in floating point.
- Embeddings and activations are quantized with per-tensor asymmetric quantization. 
	- The only exception is when two activation tensors are multiplied (present in the multi-head self-attention blocks), in which case one of the activations is quantized using symmetric quantization.
- Weights are quantized with per-tensor symmetric quantization.


