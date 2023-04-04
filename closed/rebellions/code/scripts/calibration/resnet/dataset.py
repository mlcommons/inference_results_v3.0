import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader


class PyTorchImageFolder:
    def __init__(self, image_list, resize_size=256, image_size=224, **ifkwargs) -> None:

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.imgs = image_list

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        imgp = self.imgs[index]
        img = default_loader(imgp)
        img = self.transforms(img)
        return img, 0
