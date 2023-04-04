import torch
from transformers import BertTokenizer
from create_squad_data import read_squad_examples, convert_examples_to_features

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-d', '--dataset_dir', help='Dataset directory',
        action="store", type="string", dest="dataset_dir")
parser.add_option('-m', '--model_dir', help='Model directory',
        action="store", type="string", dest="model_dir")
parser.add_option('-o', '--output', help='Output file name',
        action="store", type="string", dest="dataset_name", default='dataset.pt')

class SQuAD_DATASET():
    def __init__(self, eval_set, vocab, total_count=None, perf_count=None):
        print("Loading SQuAD dev...")

        eval_samples = read_squad_examples(
            input_file=eval_set,
            is_training=False)

        eval_features = []
        convert_examples_to_features(
            examples=eval_samples,
            tokenizer=BertTokenizer(vocab),
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            output_fn=lambda x:eval_features.append(x))

        self.eval_samples = eval_samples
        self.eval_features = eval_features
        self.total_count = len(eval_features) if total_count is None else total_count
        self.perf_count = self.total_count if perf_count is None else perf_count


if __name__ == '__main__':
    (options, args) = parser.parse_args()

    if options.model_dir is None or options.dataset_dir is None:
        parser.print_help()
        parser.exit(1)

    dev_v1 = options.dataset_dir + '/dev-v1.1.json'
    vocab_f = options.model_dir + '/vocab.txt'

    dataset = SQuAD_DATASET(eval_set = dev_v1, vocab = vocab_f)

    features = dataset.eval_features

    torch_input_ids = [torch.tensor(f.input_ids, dtype = torch.int32) for f in features]
    torch_input_mask = [torch.tensor(f.input_mask, dtype = torch.int32) for f in features]
    torch_type_ids = [torch.tensor(f.segment_ids, dtype = torch.int32) for f in features]

    dataset = {'input_ids_samples':torch_input_ids,
            'input_mask_samples':torch_input_mask,
            'segment_ids_samples':torch_type_ids}

    torch.save(dataset, options.dataset_name)
