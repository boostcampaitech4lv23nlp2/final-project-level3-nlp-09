from torchvision.transforms import (
    Resize, ToTensor,
    RandomResizedCrop,
    Normalize, Compose,
)


def clip():
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2023, 0.1994, 0.2010]

    train_tfms = Compose([
        Resize(size=(224, 224)),
        RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    valid_tfms = Compose([
        Resize(size=(224, 224)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    return train_tfms, valid_tfms
