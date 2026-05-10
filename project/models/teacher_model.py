import torch
import torch.nn as nn
import config

def teacher_model():
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    model = model.to(config.DEVICE)
    return model