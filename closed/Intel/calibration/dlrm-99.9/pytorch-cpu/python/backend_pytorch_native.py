"""
pytoch native backend for dlrm
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import backend
import torch.nn as nn
from dlrm_model import DLRM_Net, model_util
import numpy as np

class BackendPytorchNative(backend.Backend):
    def __init__(self, m_spa, ln_emb, ln_bot, ln_top, device, use_gpu=False, use_ipex=False, use_bf16=False, mini_batch_size=1, random_init=False):
        super(BackendPytorchNative, self).__init__()
        self.sess = None
        self.model = None

        self.m_spa = m_spa
        self.ln_emb = ln_emb
        self.ln_bot = ln_bot
        self.ln_top = ln_top

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.use_ipex = use_ipex
        self.use_bf16 = use_bf16
        self.device = device

        self.random_init = random_init

        ngpus = torch.cuda.device_count() if self.use_gpu else -1
        self.ndevices = min(ngpus, mini_batch_size, ln_emb.size)
        if self.use_gpu:
            print("Using {} GPU(s)...".format(ngpus))
        elif self.use_ipex:
            print("Using IPEX...")
        else:
            print("Using CPU...")

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native-dlrm"

    def random_init_emb(self, ln, emb):
        for i, l in enumerate(emb):
            n = ln[i]
            m = self.m_spa
            w = np.random.uniform(low=-np.sqrt(1 / float(n)),
                                  high=np.sqrt(1 / float(n)),
                                  size=(n, m)).astype(np.float32)
            l.weight.data = torch.tensor(w, requires_grad=True)

    def random_init_mlp(self, ln, layers):
        j = 0
        for l in layers:
            if isinstance(l, nn.Linear):
                n = ln[j]
                m = ln[j + 1]
                w = np.random.normal(0.0,
                                     np.sqrt(2 / float(n + m)),
                                     size=(m, n)).astype(np.float32)
                l.weight.data = torch.tensor(w, requires_grad=True)
                b = np.random.normal(0.0,
                                     np.sqrt(1 / float(m)),
                                     size=m).astype(np.float32)
                l.bias.data = torch.tensor(b, requires_grad=True)
                j += 1

    def load(self, model_path, inputs=None, outputs=None):
        # debug prints
        # print(model_path, inputs, outputs)

        dlrm = DLRM_Net(
            self.m_spa,
            self.ln_emb,
            self.ln_bot,
            self.ln_top,
            arch_interaction_op="dot",
            arch_interaction_itself=False,
            sigmoid_bot=-1,
            sigmoid_top=self.ln_top.size - 2,
            sync_dense_params=True,
            loss_threshold=0.0,
            ndevices=self.ndevices,
            qr_flag=False,
            qr_operation=None,
            qr_collisions=None,
            qr_threshold=None,
            md_flag=False,
            md_threshold=None,
            bf16=self.use_bf16,
            use_ipex=self.use_ipex,
        )
        if self.use_gpu:
            dlrm = dlrm.to(self.device)  # .cuda()
            if dlrm.ndevices > 1:
                dlrm.emb_l = dlrm.create_emb(self.m_spa, self.ln_emb)

        if self.random_init:
            self.random_init_emb(self.ln_emb, dlrm.emb_l)
            self.random_init_mlp(self.ln_bot, dlrm.bot_l)
            self.random_init_mlp(self.ln_top, dlrm.top_l)
        else:
            if self.use_gpu:
                if dlrm.ndevices > 1:
                    # NOTE: when targeting inference on multiple GPUs,
                    # load the model as is on CPU or GPU, with the move
                    # to multiple GPUs to be done in parallel_forward
                    ld_model = torch.load(model_path)
                else:
                    # NOTE: when targeting inference on single GPU,
                    # note that the call to .to(device) has already happened
                    ld_model = torch.load(
                        model_path,
                        map_location=torch.device('cuda')
                        # map_location=lambda storage, loc: storage.cuda(0)
                    )
            else:
                # when targeting inference on CPU
                ld_model = torch.load(model_path, map_location=torch.device('cpu'))
                # debug print
                # print(ld_model)
                dlrm.load_state_dict(ld_model["state_dict"])
        if self.use_ipex:
            dlrm = model_util(dlrm, self.use_bf16, self.device)
        self.model = dlrm

        # find inputs from the model if not passed in by config
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = []
            initializers = set()
            for i in self.model.graph.initializer:
                initializers.add(i.name)
            for i in self.model.graph.input:
                if i.name not in initializers:
                    self.inputs.append(i.name)

        # find outputs from the model if not passed in by config
        if outputs:
            self.outputs = outputs
        else:
            self.outputs = []
            for i in self.model.graph.output:
                self.outputs.append(i.name)
        self.model.emb_l_w = [e.weight for e in self.model.emb_l]
        # debug print
        # for e in self.ln_emb:
        #     print('Embedding', type(e), e)
        return self.model

def get_backend(backend, dataset, device, max_ind_range, data_sub_sample_rate, use_gpu, use_ipex, use_bf16):
    if backend == "pytorch-native":
        # NOTE: pass model parameters here, the following options are available
        if dataset == "kaggle":
            # 1. Criteo Kaggle Display Advertisement Challenge Dataset (see ./bench/dlrm_s_criteo_kaggle.sh)
            backend = BackendPytorchNative(
                m_spa=16,
                ln_emb=np.array([1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572]),
                ln_bot=np.array([13,512,256,64,16]),
                ln_top=np.array([367,512,256,1]),
                device=device,
                use_gpu=use_gpu,
                use_ipex=use_ipex,
                use_bf16=use_bf16
            )
        elif dataset == "terabyte":
            if max_ind_range == 10000000:
                # 2. Criteo Terabyte (see ./bench/dlrm_s_criteo_terabyte.sh [--sub-sample=0.875] --max-in-range=10000000)
                backend = BackendPytorchNative(
                    m_spa=64,
                    ln_emb=np.array([9980333,36084,17217,7378,20134,3,7112,1442,61, 9758201,1333352,313829,10,2208,11156,122,4,970,14, 9994222, 7267859, 9946608,415421,12420,101, 36]),
                    ln_bot=np.array([13,512,256,64]),
                    ln_top=np.array([415,512,512,256,1]),
                    device=device,
                    use_gpu=use_gpu,
                    use_ipex=use_ipex,
                    use_bf16=use_bf16
                )
            elif max_ind_range == 40000000:
                # 3. Criteo Terabyte MLPerf training (see ./bench/run_and_time.sh --max-in-range=40000000)
                backend = BackendPytorchNative(
                    m_spa=128,
                    ln_emb=np.array([39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36]),
                    ln_bot=np.array([13,512,256,128]),
                    ln_top=np.array([479,1024,1024,512,256,1]),
                    device=device,
                    use_gpu=use_gpu,
                    use_ipex=use_ipex,
                    use_bf16=use_bf16
                )
            else:
                ln_emb = [39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36]
                ln_emb = np.array([x if x < max_ind_range else max_ind_range for x in ln_emb])
                backend = BackendPytorchNative(
                    m_spa=128,
                    ln_emb=ln_emb,
                    ln_bot=np.array([13,512,256,128]),
                    ln_top=np.array([479,1024,1024,512,256,1]),
                    device=device,
                    use_gpu=use_gpu,
                    use_ipex=use_ipex,
                    use_bf16=use_bf16,
                    random_init=True
                )
        else:
            raise ValueError("only kaggle|terabyte dataset options are supported")

    else:
        raise ValueError("unknown backend: " + backend)
    return backend
