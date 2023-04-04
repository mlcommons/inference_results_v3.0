import sys
import os
import torch
import numpy as np
#from backend_pytorch_native import get_backend
model_path = sys.argv[1] # '/DataDisk/dlrm_dataset/model/dlrm_terabyte.pytorch'
dump_path = sys.argv[2] # '/DataDisk_3/syk/dlrm_model.npz'
#backend = get_backend('pytorch-native', 'terabyte', 'xpu', 40000000, 0.0,
#                      False, False, False, False)
#model = backend.load(model_path,
#                     'continuous and categorical features',
#                     'probability')

weights = torch.load(model_path)

param = {}
for k, v in weights['state_dict'].items():
    w = v.detach().cpu().numpy()
    if k == 'bot_l.0.weight':
        w_padded = np.zeros([512, 32], dtype=np.float32)
        w_padded[:, 0:13] = w[:, :]
        w = w_padded
    if k == 'top_l.0.weight':
        w_padded = np.zeros([1024, 512], dtype=np.float32)
        w_padded[:, 0:479] = w[:, :]
        w = w_padded
    param[k] = w
np.savez(dump_path, **param)
