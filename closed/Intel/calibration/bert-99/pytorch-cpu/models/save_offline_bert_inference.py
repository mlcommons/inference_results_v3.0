import sys
sys.path.append('..')
import os.path

import torch
import transformers
from transformers import BertTokenizer

from quant_modules import propagate_quantizer
from modeling_half_bert import BertForQuestionAnswering

from optparse import OptionParser

torch._C._jit_set_profiling_mode(False)
transformers.logging.set_verbosity_info()

parser = OptionParser()
parser.add_option('-m', '--model_dir', help='Model directory',
        action="store", type="string", dest="model_dir")
parser.add_option('-o', '--output', help='Output file name',
        action="store", type="string", dest="model_name", default='../bert.pt')

if __name__ == '__main__':
    (options, args) = parser.parse_args()

    model_dir = options.model_dir
    if model_dir is None:
        parser.print_help()
        parser.exit(1)

    model_file = model_dir + '/pytorch_model.bin'
    config_file = model_dir + '/config.json'
    vocab_file = model_dir + '/vocab.txt'

    tokenizer = BertTokenizer(vocab_file)
    model = BertForQuestionAnswering.from_pretrained(model_file, config=config_file)
    model = propagate_quantizer(model)
    jmodel = torch.jit.script(model)
    fmodel = torch.jit.freeze(jmodel)
    torch._C._jit_pass_constant_propagation(fmodel.graph)

    model_name = options.model_name
    torch.jit.save(fmodel, model_name)
