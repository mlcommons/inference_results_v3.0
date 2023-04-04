# MLPerf Inference v3.0 - Qualcomm - Calibration Details

## Qualcomm Cloud AI 100

We use regular profile-guided post-training quantization by applying the Qualcomm Cloud AI toolchain.
This involves couple of steps as described below.

### Step 1: Profile generation

We pass a set of calibration samples through the neural network executed in single-precision floating-point to obtain a profile of every activation tensor of the network. The profile consists of information such as dynamic range and histogram of activations with `N` bins for every activation tensor in the neural network. It also consists of dynamic range and histogram of weights for Convolution and Fully Connected layers. The `N` bins histogram for each activation starts with an initial `MIN` and `MAX` of the input tensor and as the `MIN` and `MAX` change with every new input tensor, the bins centers are rescaled, and bin counts are redistributed. Number of histogram bins `N = 512` for this submission.

### Step 2: Quantization

#### Quantization of activations
Use the generated profile information from above step to compute a quantization `[scale, offset]` for every activation tensor. Depending on the need to represent the activation tensor symmetrically around `0` or asymmetric the scale and offset computation differ. Be it symmetric or asymmetric, `0` is always considered to be part of the dynamic range.
The `MIN-MAX` range of the activation tensors can be clipped to a smaller range so that the `2` to the power of `bitwidth` levels are used to represent more useful part of the histogram while the outliers beyond the clipped `MIN-MAX` ranges would be counted in the first and last bins. The criteria based on which the clipping levels are chosen may aim to minimize the mean square error (MSE) between the original histogram vs. clipped histogram or aim to minimize KL Divergence between the original histogram vs. clipped histogram. Sometimes the clipped levels can be chosen as a percentile of the original histogram with any percentile value that may help emphasize the most useful portion of the histogram leaving out the outliers. The `bitwidth = 8` bits for this submission.

#### Quantization of weights
Quantization scale and offset for weight tensors can be derived on a per-weight tensor basis or on a per-channel basis for Convolution layers / per-row basis for Fully Connected layers. For some models, per-channel or per-row `[scale, offset]` are used, while for some models (wherever applicable) per-tensor `[scale, offset]` are used. The full `MIN-MAX` ranges of the weights tensor are used without any clipping.
