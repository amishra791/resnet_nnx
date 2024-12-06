import torchvision
from transform import IMAGENET_TRANSFORM
import torch
from torch.utils.data import DataLoader, Subset

IMAGENET_HOME = "/mnt/disks/imagenet"

def get_dataset():
    train_dataset = torchvision.datasets.ImageNet(IMAGENET_HOME, split="train", transform=IMAGENET_TRANSFORM)
    val_dataset = torchvision.datasets.ImageNet(IMAGENET_HOME, split="val", transform=IMAGENET_TRANSFORM)

    return {
        'train': train_dataset,
        'val': val_dataset
    }

def get_dataloaders(batch_size):
    datasets = get_dataset()

    # shuffle during training since images are ordered by class
    train_dataloader = DataLoader(dataset=datasets['train'], batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=datasets['val'], batch_size=batch_size)

    return train_dataloader, val_dataloader
