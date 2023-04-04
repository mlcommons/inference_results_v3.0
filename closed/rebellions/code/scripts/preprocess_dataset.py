# From mlperf root directory
# python code/scripts/preprocess_dataset.py squad -p ~/preprocessed/
# python code/scripts/preprocess_dataset.py imagenet -p ~/preprocessed/

from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("type", type=str, choices=["imagenet", "squad"])
parser.add_argument("-p", "--preprocessed-path", type=str, default="preprocessed/")
args = parser.parse_args()

scratch_path = os.path.abspath(os.path.join("scratch", "preprocessed", args.type))
preprocessed_path = os.path.abspath(os.path.join(args.preprocessed_path, args.type))

os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
os.makedirs(preprocessed_path, exist_ok=True)

if os.path.exists(scratch_path):
    os.unlink(scratch_path)
os.symlink(preprocessed_path, scratch_path)


if args.type == "imagenet":
    from imagenet import *

    preprocess(
        data_root="scratch/datasets/ILSVRC2012",
        cache_dir=preprocessed_path,
        scale_factor=48.106057336853475,
    )
elif args.type == "squad":
    from squad import *

    tokenize(
        data_file="scratch/datasets/squad_v1.1/dev-v1.1.json",
        vocab_file="scratch/datasets/squad_v1.1/vocab.txt",
        cache_file=os.path.join(preprocessed_path, "squad.npy"),
    )
    # squad.npy : (10833, 3, 384) np.int32, [input_ids, segment_ids, input_mask]

else:
    raise NotImplementedError()
