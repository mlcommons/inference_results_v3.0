The folder has 2 python scripts, compress_to15 and compress_to16. These scripts are used to generate the compressed pre-processed data from the uncompressed pre-processed data. The uncompressed data is 89M x 26 numpy array. We can compress this to either 89M x 15 or 89M x 16. The first option results in a smaller memory copy, while the second option helps the decompression kernel run faster due to better memory alignment but at the cost of more memory copy. The user can decide which of the two options he prefers, and choose between the two scripts.

The uncompressed numpy array can be found at:
/home/mlperf_inference_data/preprocessed_data/criteo/full_recalib/categorical_int32.npy

Before running compression, this uncompressed numpy array should be copied to the same folder the Python script is. The python script gives an output file named "categorical_int32_compressed.npy" in the same folder. This output numpy array is either 89M x 15 or 89M x 16 depending on which script of the two you run.

NOTE: The uncompressed file should be in the same folder as the script before running, and should be strictly named "categorical_int32.npy".
