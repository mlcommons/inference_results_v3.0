# MLPerf Inference Calibration and Quantization Details
---

## Moffett Quantization Rules

- Quantization is symmetric for both weight and activation, where weights have per-channel scales and activation has per-layer scale, as follows. 

- For ResNet50 and RetinaNet, the quantization is applied to all tensors in the forward inference pass, and use KL-divergence method to calibrate activation and weight tensors with a calibration set of 1000 images from ImageNet training dataset. 

- For BERT-large, the quantization is only applied to the input and output tensors of matrix multiplication. Use min-max method to calibrate activation and weight tensors with a calibration set of only a batch of instances from SQUAD training set.

### Entropy calibration (used in RN50, RetinaNet)



- During calibration we need to determine (i) the scale of activatation layer and (ii) the scale of each output channel of a weight as follows 
	- For each activation tensor, we use a distinct dynamic range that applies across the entire tensor. We invoke the model on a set of representative inputs in FP32 precision, and create a per-tensor histogram of absolute values. The histogram initially uses 1024 equal-range bins whose range is set by the initial image batch, but dynamically resizes by doubling the number of bins as necessary to accommodate the range of subsequent batches. Call this histogram, which has [power-of-2] bins, where all data elements are guaranteed to fall into one of the bins, the starting histogram.
		- for each bin B in the starting histogram, we compute a divergence value as follows
			- Create a truncated histogram where each bin has the same range and count as the original, except that all elements in bins beyond B are considered to be in B, and all bins beyond B are removed.
			- Create a coarse histogram by discretizing the truncated histogram into 127 bins of equal range between 0 and the midpoint of B, placing all elements in the final bin of the truncated histogram into the final bin of the coarse histogram.
			- Compute the KL-divergence between the distributions represented by the coarse histogram and the truncated histogram.
			- The dynamic range chosen is the center of the bin which minimizes divergence.
	- For weights, Dynamic range values are generally per-channel. We find the maximum absolute value t of any element of the channel or tensor, and the dynamic range is then [-t,t].

### min-max calibration (used in BERT-large)  

- During calibration we need to determine (i) the scale of activatation layer and (ii) the scale of each output channel of a weight as follows
	- For activations, We find the maximum absolute value t of any element of the tensor, and the dynamic range is then [-t,t].
	- For weights,  We find the maximum absolute value t of any element of the output channel, and the dynamic range is then [-t,t].

### Additional Details

- For BERT-large, a regularization term is added to the objective function during re-training to minimize the quantization error of each activation layer's output tensor.