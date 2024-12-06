import torchvision
from torchvision.transforms import v2
import torch


IMAGENET_TRANSFORM = v2.Compose([
    torchvision.transforms.ToTensor(),
    v2.RandomResize(min_size=256, max_size=480),
    v2.RandomCrop(size=224),
    # normalization constants taken from https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/imagenet.lua
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # make tensor of shape HWC
    torchvision.transforms.Lambda(lambda images: torch.permute(images, (1, 2, 0)))
])