import torch
import transformers
import torch.autograd.profiler as profiler

from transformers import BertTokenizer, BertForQuestionAnswering
from optparse import OptionParser

torch._C._jit_set_profiling_mode(False)
transformers.logging.set_verbosity_info()

parser = OptionParser()
parser.add_option('-m', '--model_dir', help='Model directory',
        action="store", type="string", dest="model_dir")
parser.add_option('-o', '--output', help='Output file name',
        action="store", type="string", dest="model_name", default='dqbert.pt')

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
    model_dq = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    jitted_model = torch.jit.script(model_dq)
    frozen_model = torch.jit.freeze(jitted_model)

    torch.jit.save(frozen_model, options.model_name)
