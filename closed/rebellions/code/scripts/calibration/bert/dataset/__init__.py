from json import load
from pathlib import Path
from .tokenization import FullTokenizer
from .create_squad_data import convert_examples_to_features_calib, read_squad_examples


class TFSQuAD:
    def __init__(self, root) -> None:
        self.root = Path(root)
        # with open('/srv/data/squad_v1.1/dev-v1.1.json') as f:
        with (self.root / "dev-v1.1.json").open() as f:
            input_data = load(f)["data"]
            input_data = input_data
        self.features = []
        self.tokenizer = FullTokenizer(str(root / "vocab.txt"))

        # for MLPerf
        calib_examples_full = read_squad_examples(
                                input_data=input_data, is_training=False, version_2_with_negative=False)
        convert_examples_to_features_calib(
            examples = calib_examples_full,
            tokenizer = self.tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            output_fn=lambda x: self.features.append(x),
            verbose_logging=False,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if index >= len(self):  # For python <= 3.6
            raise StopIteration()
        return self.features[index]
