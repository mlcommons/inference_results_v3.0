import logging
import os
import sys

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0])
import torch

from _C import *


LOG_LEVEL = (
    int(os.environ["RNNT_LOG_LEVEL"])
    if "RNNT_LOG_LEVEL" in os.environ
    else logging.INFO
)
LOG_FORMAT = logging.Formatter("[%(filename)s:%(lineno)d %(levelname)s] %(message)s")
logger = logging.getLogger("RNNTLogger")
logger.setLevel(LOG_LEVEL)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(LOG_FORMAT)
logger.addHandler(stream_handler)

labels = [
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "'",
]


def seq_to_sen(seq, seq_lens):
    sen = "".join([labels[seq[idx]] for idx in range(seq_lens)])
    return sen


def migrate_state_dict(model, split_fc1=True):
    state_dict = model["state_dict"] if "state_dict" in model else model
    migrated_state_dict = {}
    for key, value in state_dict.items():
        if key == "joint_net.0.weight" and split_fc1:
            migrated_state_dict["joint.linear1_trans.weight"] = value[:, :1024]
            migrated_state_dict["joint.linear1_pred.weight"] = value[:, 1024:]
            continue
        if key == "joint_net.0.bias" and split_fc1:
            migrated_state_dict["joint.linear1_trans.bias"] = torch.zeros(512)
            migrated_state_dict["joint.linear1_pred.bias"] = value
        key = key.replace("encoder.pre_rnn.lstm", "transcription.pre_rnn")
        key = key.replace("encoder.post_rnn.lstm", "transcription.post_rnn")
        key = key.replace("dec_rnn.lstm", "pred_rnn")
        key = key.replace("joint_net.0", "joint.linear1")
        key = key.replace("joint_net.3", "joint.linear2")
        migrated_state_dict[key] = value
    if "audio_preprocessor.featurizer.fb" in migrated_state_dict.keys():
        del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    if "audio_preprocessor.featurizer.window" in migrated_state_dict.keys():
        del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict


def jit_module(module, to_freeze=True):
    jmodule = torch.jit.script(module)
    if to_freeze:
        fmodule = torch.jit._recursive.wrap_cpp_module(
            torch._C._freeze_module(jmodule._c)
        )
        # module = torch.jit.freeze(module)
        torch._C._jit_pass_constant_propagation(fmodule.graph)
        return fmodule
    else:
        return jmodule


def jit_model(model):
    for name, layer in model.transcription.named_children():
        if isinstance(layer, torch.nn.LSTM):
            for sub_name, sub_layer in layer.named_children():
                setattr(layer, f"{sub_name}", jit_module(sub_layer))
            setattr(model.transcription, f"{name}", jit_module(layer, False))
        else:
            setattr(model.transcription, f"{name}", jit_module(layer))
    model.transcription = jit_module(model.transcription, False)
    model.prediction = jit_module(model.prediction)
    model.joint = jit_module(model.joint)
    model.update = jit_module(model.update)
    model = jit_module(model, False)
    return model
