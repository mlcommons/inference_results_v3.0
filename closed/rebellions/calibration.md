# MLPerf  Inference  v3.0  -  Rebellions  Quantization  Details

We adopt a symmetric uniform quantization for both weights and activations as described below:
```math
Q(w)= \Delta  \cdot \text{clip}\biggl(\text{round}\biggl(\dfrac{w}{\Delta}\biggr) , -128, 127\biggr) = \Delta \cdot z,
```
where $\Delta$ is the scale factor, $w$ represents a set of floating-point weights/activations.

The following calibration algorithms are used to determine the scale factor, depending on target benchmark area (vision and language):

1.L2 Error Minimization Calibration:
    
To select an appropriate scale factor $\Delta$, we apply the L2 error minimization criteria. The quantization error $E$ is represented as:
```math
E = \dfrac1 2\sum\limits_{i=1}^N\bigl(Q(w_{i})-w_{i} \bigr)^2 = \dfrac1 2\sum\limits_{i=1}^N\bigl(\Delta \cdot z_{i}-w_{i} \bigr)^2,
```
where $N$ is the number of weights/activations in each layer, $w_{i}$ is the $i$-th weight/activation value in a floating-point format, and $z_{i}$ is the integer membership of $w_{i}$, which is determined by the quantizer.
The quantization error $E$ can be minimized iteratively through the following two-step computations:
```math
\boldsymbol{z}^{(t)} =  \text{clip}\biggl(\text{round}\biggl(\dfrac{\boldsymbol{w}}{\Delta^{(t-1)}}\biggr) , -128, 127\biggr),
```
```math
\Delta^{(t)} =\arg\min_{\Delta}\operatorname{\textit{E}}(\boldsymbol{w}, \boldsymbol{z}^{(t)}, \Delta) = \dfrac {\sum\limits_{i=1}^N w_{i} \cdot z^{(t)}_{i}  }{\sum\limits_{i=1}^N\bigl(z_{i}^{(t)} \bigr)^2},
```
where the superscript $(t)$ denotes the iteration step. The iteration terminates when $\Delta^{(t)}$ has converged.


2.Min/Max Calibration:

The scale factor $\Delta$ of the min/max calibration is determined with the following equation:
```math
\Delta  =  \dfrac{\text{max}(\text{abs}(w))}{2^{(b-1)}-1},
```
where $b$ is the quantization bit-width, i.e. 8 in our case.


3.Percentile Calibration:

The scale factor $\Delta$ of the percentile calibration is determined with the following equation:
```math
\Delta  =  \dfrac{\text{percentile}(\text{abs}(w'), \alpha)}{2^{(b-1)}-1},
```
where $\text{percentile}(\cdot)$ is a function that returns the $\alpha$-th percentile of the array elements, $w'$ is a sorted set of weights/activations from the smallest to the largest, $\alpha$ is the percentile between 0 and 100, and $b$ is the quantization bit-width, 8 in our case.


4.Fixed Scale Calibration:

The scale factor $\Delta$ of the fixed scale calibration is determined by the following equation:

```math
\Delta  =  \dfrac{1}{2^{(b-1)}-1},
```
where $b$ is the quantization bit-width, 8 in our case.

## Activations
We use a per-tensor based post training quantization for activations. All hidden activations of the input calibration data are used for percentile calibration, with or without fixed scale method according to target models. Here is the details of the method for each category:

- Vision: percentile calibration with $\alpha$ = 99.999%
- Language: percentile calibration with $\alpha$ = 99.999%, and a fixed scale factor of 1/127 for the outputs of softmax layers

## Weights
We use a per-channel based post training quantization for weights. Two different calibration algorithms (L2 error minimization and Min/Max calibrations) are used according to target area as below:

- Vision: Min/Max
- Language: L2 error minimization
