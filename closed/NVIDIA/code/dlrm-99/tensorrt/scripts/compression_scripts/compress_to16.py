# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

categ_inputs = np.load('categorical_int32.npy')

num_examples = categ_inputs.shape[0]
compressed_inputs = np.empty([num_examples, 16], dtype=np.int32)

for i in range(num_examples):
    compressed_inputs[i][0] = categ_inputs[i][0]
    compressed_inputs[i][1] = ((categ_inputs[i][1] << 16) + categ_inputs[i][2]).astype(np.int32)
    compressed_inputs[i][2] = ((categ_inputs[i][3] << 16) + categ_inputs[i][4]).astype(np.int32)
    compressed_inputs[i][3] = ((categ_inputs[i][6] << 16) + categ_inputs[i][7]).astype(np.int32)
    compressed_inputs[i][4] = categ_inputs[i][9]
    compressed_inputs[i][5] = categ_inputs[i][10]
    compressed_inputs[i][6] = categ_inputs[i][11]
    compressed_inputs[i][7] = ((categ_inputs[i][13] << 16) + categ_inputs[i][14]).astype(np.int32)
    compressed_inputs[i][8] = ((categ_inputs[i][5] << 24) + (categ_inputs[i][8] << 16) + (categ_inputs[i][12] << 8) + categ_inputs[i][15]).astype(np.int32)
    compressed_inputs[i][9] = ((categ_inputs[i][16] << 24) + (categ_inputs[i][17] << 8) + categ_inputs[i][18]).astype(np.int32)
    compressed_inputs[i][10] = categ_inputs[i][19]
    compressed_inputs[i][11] = categ_inputs[i][20]
    compressed_inputs[i][12] = categ_inputs[i][21]
    compressed_inputs[i][13] = categ_inputs[i][22]
    compressed_inputs[i][14] = ((categ_inputs[i][23] << 16) + (categ_inputs[i][24] << 8) + categ_inputs[i][25]).astype(np.int32)
    compressed_inputs[i][15] = 0

print(compressed_inputs[0])
np.save('categorical_int32_compressed.npy', compressed_inputs)
