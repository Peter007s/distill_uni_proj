import torch.nn as nn
from torchvision import models
import config

def teacher_model():
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(2048, config.NUM_CLASSES)
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    model = model.to(config.DEVICE)
    return model