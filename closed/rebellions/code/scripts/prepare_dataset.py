from argparse import ArgumentParser
from pathlib import Path
import typing

parser = ArgumentParser()
parser.add_argument("type", type=str, choices=["imagenet", "squad"])
parser.add_argument("-p", "--dataset-path", type=Path, default=None)
args = parser.parse_args()


dataset_path: typing.Optional[Path] = args.dataset_path

if args.type == "imagenet":
    DATASET_SCRATCH_PATH = Path("scratch/datasets/ILSVRC2012")
elif args.type == "squad":
    DATASET_SCRATCH_PATH = Path("scratch/datasets/squad_v1.1")
else:
    raise NotImplementedError()

DATASET_SCRATCH_PATH.parent.mkdir(parents=True, exist_ok=True)

if dataset_path and dataset_path.is_dir():
    DATASET_SCRATCH_PATH.unlink(missing_ok=True)
    DATASET_SCRATCH_PATH.symlink_to(dataset_path)
else:
    raise NotImplementedError()
