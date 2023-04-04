import os
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from tqdm import tqdm


def fake_quantize(torch_img: torch.Tensor, scale_factor: float):
    # torch_img = torch_img.to(torch.float16)
    torch_img *= scale_factor
    # torch_img = torch_img.to(torch.float32)
    torch_img.round_()
    torch.clamp_(torch_img, min=-128.0, max=127.0)
    return torch_img


@torch.no_grad()
def preprocess_one(image_path: str, cache_dir: str, scale_factor: float):
    cache_path = os.path.join(cache_dir, os.path.basename(image_path) + ".npy")
    if os.path.exists(cache_path):
        return
    resize_size = 256
    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    image = default_loader(image_path)
    image = transform(image)
    image = fake_quantize(image, scale_factor=scale_factor)
    image = torch.cat((image, torch.zeros(1, 224, 224, dtype=torch.float32)), dim=0)
    image.unsqueeze_(0)
    image = image.to(torch.int8)
    image = image.permute(0, 2, 3, 1)
    image = image.numpy()
    np.save(cache_path, image)


def preprocess(data_root: str, cache_dir: str, scale_factor: float):
    os.makedirs(cache_dir, exist_ok=True)
    image_paths = [
        os.path.join(data_root, "ILSVRC2012_img_val", f"ILSVRC2012_val_{i + 1:08d}.JPEG")
        for i in range(50000)
    ]
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(preprocess_one)(image_path, cache_dir, scale_factor)
        for image_path in tqdm(image_paths)
    )
