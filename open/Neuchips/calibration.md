# **NEUCHIPS MLPerf™ Quantization**
In MLPerf™ v3.0, we use dynamic-range symmetric quantization aware training for quantizing weights, activations, and embeddings from FP32 to int8/uint8.

## **Weights**
We use layer-wise and table-wise quantization for MLP and Embedding separately. The quantization steps are as below:

* Step1: find the maximum absolute value of weights/embeddings for each layer/table. 
* Step2: divide the maximum absolute value by 128 to get the quantization scale
* Step3: quantize weights/embeddings to int8 with the quantization scale.

## **Activations**
For activations, we use layer-wise quantization. The detail quantization steps are as below:

* Step1: find the maximum absolute value of activations from each layer. 
* Step2: divide the maximum absolute value by 128 to get the quantization scale.
* Step3: quantize activations to int8 with the quantization scale.
