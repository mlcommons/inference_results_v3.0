# **NEUCHIPS MLPerf™ Quantization**
In MLPerf™ v3.0, we use dynamic-range symmetric quantization for quantize weights, activations, and embeddings from FP32 to int8/uint8.

## **Weights**
We use layer-wise and table-wise quantization for MLP and Embedding separately. The quantization steps are as below:

* Step1: find Min/Max of weights for each layer/table. 
* Step2: find the scaling factor S, which S is maximum(Min/-128, Max/127). 
* Step3: quantize weights to int8 by S. 

## **Activations**
For activations, we use layer-wise quantization. The detail quantization steps are as below:

* Step1: find Min/Max of activations for each layer.
* Step2: find the scaling factor S, which S is maximum(Min\*M/-128, Max\*M/127). M is a hyperparameter in the range [0.8, 1.0].
* Step3: compute all results of calibration set for each M, and find the best M so that the scaling factor S is decided.
* Step4: quantize activations to uint8 by S if there is an activation function.
         otherwise, quantize activations to int8 by S.
         the numbers that are larger than 127 or less than -128 are saturated.
