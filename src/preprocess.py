from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

OPENAI_DATASET_MEAN = (0.0026, 0.0022, 0.0018)
OPENAI_DATASET_STD = (0.0017, 0.0016, 0.0016)


class ResizeMaxSize(nn.Module):
    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn="max", fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == "min" else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert("RGB")


def image_transform(
    image_size: int,
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    resize_longest_max: bool = False,
    fill_color: int = 0,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        return Compose(
            [
                RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )
    else:
        if resize_longest_max:
            transforms = [ResizeMaxSize(image_size, fill=fill_color)]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend(
            [
                _convert_to_rgb,
                ToTensor(),
                # normalize,
            ]
        )
        return Compose(transforms)


def food_transform():
    mean = [0.4482, 0.3830, 0.3044]
    std = [0.2886, 0.2768, 0.2754]

    train_tfms = Compose(
        [
            Resize(size=(224, 224)),
            RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0)),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    valid_tfms = Compose(
        [
            Resize(size=(224, 224)),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    return train_tfms, valid_tfms
