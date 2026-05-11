import torch
import torch.nn as nn
import config
from models.custom_cnn import CifarSmall

def create_student(name):
    if name == 'resnet20':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False)
        
    elif name == 'resnet32':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=False)

    elif name == 'custom':
        model = CifarSmall(32, 10)
    model = model.to(config.DEVICE)
    return model