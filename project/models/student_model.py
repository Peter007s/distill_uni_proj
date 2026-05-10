import torch
import torch.nn as nn
import config

def create_student(name):
    if name == 'resnet20':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False)
        
    elif name == 'resnet32':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=False)
    model = model.to(config.DEVICE)
    return model