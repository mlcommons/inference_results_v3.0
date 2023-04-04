"""
pytoch native backend for dlrm
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import backend
import time

def reshape_weight(input):
    output = torch.empty(size = input.shape, dtype=torch.int8).reshape((-1,64))
    k = 0
    for i in range(0, input.shape[0], 16):
        for j in range(0, input.shape[1], 4):
            output[k] = input[i:i+16, j:j+4].flatten()
            k+=1
    return output

class BackendPytorch_NEUCHIPS_DLRM(backend.Backend):
    def __init__(self, use_gpu=False, mini_batch_size=1):
        super(BackendPytorch_NEUCHIPS_DLRM, self).__init__()
        self.sess = None
        self.model = None

        

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda:0" if self.use_gpu else "cpu"

        ngpus = torch.cuda.device_count() if self.use_gpu else -1
        self.ndevices = min(ngpus, mini_batch_size)
        if self.use_gpu:
            print("Using {} GPU(s)...".format(ngpus))
        else:
            print("Using CPU...")

        torch.classes.load_library("./NEUCHIPS_DLRM.so")
        # libdlrm_avx512_2x2_MaxBatch12_82K.so
        # 0213_84k_libdlrm_avx512.so (69540 QPS)

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-NEUCHIPS-dlrm"

    def load(self, model_path, inputs=None, outputs=None):
        # debug prints
        # print(model_path, inputs, outputs)

        self.ld_model = torch.load("./NEUCHIPS_DLRM.pt", map_location='cpu')

        self.padded_weight_bot_layer1 = torch.nn.functional.pad(self.ld_model['state_dict']['bot_l.0.weight'], (0,3), "constant", 0)
        self.padded_weight_top_layer1 = torch.nn.functional.pad(self.ld_model['state_dict']['top_l.0.weight'], (0,1), "constant", 0)
        self.reshaped_weight_bot_layer1 = reshape_weight(self.padded_weight_bot_layer1)
        self.reshaped_weight_bot_layer2 = reshape_weight(self.ld_model['state_dict']['bot_l.2.weight'])
        self.reshaped_weight_bot_layer3 = reshape_weight(self.ld_model['state_dict']['bot_l.4.weight'])
        self.reshaped_weight_top_layer1 = reshape_weight(self.padded_weight_top_layer1)
        self.reshaped_weight_top_layer2 = reshape_weight(self.ld_model['state_dict']['top_l.2.weight'])
        self.reshaped_weight_top_layer3 = reshape_weight(self.ld_model['state_dict']['top_l.4.weight'])

        self.model = torch.classes.my_classes.DLRMClass(
            self.ld_model['state_dict']['emb_l.0.weight'],
            self.ld_model['state_dict']['emb_l.1.weight'],
            self.ld_model['state_dict']['emb_l.2.weight'],
            self.ld_model['state_dict']['emb_l.3.weight'],
            self.ld_model['state_dict']['emb_l.4.weight'],
            self.ld_model['state_dict']['emb_l.5.weight'],
            self.ld_model['state_dict']['emb_l.6.weight'],
            self.ld_model['state_dict']['emb_l.7.weight'],
            self.ld_model['state_dict']['emb_l.8.weight'],
            self.ld_model['state_dict']['emb_l.9.weight'],
            self.ld_model['state_dict']['emb_l.10.weight'],
            self.ld_model['state_dict']['emb_l.11.weight'],
            self.ld_model['state_dict']['emb_l.12.weight'],
            self.ld_model['state_dict']['emb_l.13.weight'],
            self.ld_model['state_dict']['emb_l.14.weight'],
            self.ld_model['state_dict']['emb_l.15.weight'],
            self.ld_model['state_dict']['emb_l.16.weight'],
            self.ld_model['state_dict']['emb_l.17.weight'],
            self.ld_model['state_dict']['emb_l.18.weight'],
            self.ld_model['state_dict']['emb_l.19.weight'],
            self.ld_model['state_dict']['emb_l.20.weight'],
            self.ld_model['state_dict']['emb_l.21.weight'],
            self.ld_model['state_dict']['emb_l.22.weight'],
            self.ld_model['state_dict']['emb_l.23.weight'],
            self.ld_model['state_dict']['emb_l.24.weight'],
            self.ld_model['state_dict']['emb_l.25.weight'],
            self.reshaped_weight_bot_layer1,
            self.ld_model['state_dict']['bot_l.0.bias'],
            self.reshaped_weight_bot_layer2,
            self.ld_model['state_dict']['bot_l.2.bias'],
            self.reshaped_weight_bot_layer3,
            self.ld_model['state_dict']['bot_l.4.bias'],
            self.reshaped_weight_top_layer1,
            self.ld_model['state_dict']['top_l.0.bias'],
            self.reshaped_weight_top_layer2,
            self.ld_model['state_dict']['top_l.2.bias'],
            self.reshaped_weight_top_layer3,
            self.ld_model['state_dict']['top_l.4.bias'],
            self.ld_model['state_dict']['top_l.6.weight'],
            self.ld_model['state_dict']['top_l.6.bias'],
        )
        return self

    # def predict(self, batch_dense_X, batch_lS_o, batch_lS_i, num_threads=1):
    def predict(self, batch_dense_X, batch_lS_i, num_threads=1):

        #print("batch_lS_i : ", batch_lS_i)
        
        size = batch_lS_i.shape[0]
        #print("num_threads in backend: ", num_threads)
        # result_tensor = torch.zeros((batc h_lS_i.shape[0]), dtype=torch.float32)
        result_tensor = torch.empty((batch_lS_i.shape[0]), dtype=torch.float32)
        try:
            # st = time.perf_counter()
            #print("backend predict start !!")
            self.model.dlrm_forward(size, batch_dense_X, batch_lS_i, result_tensor, num_threads)
            #print("backend predict done !!")
            # et = time.perf_counter()
            # print('{}---{}------'.format(size, et - st))
        except Exception as e:
            print('predict error:',e)
        return result_tensor

