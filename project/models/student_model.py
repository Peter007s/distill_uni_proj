import torch.nn as nn
from torchvision import models
import config

def create_student(name):
    if name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, config.NUM_CLASSES)
    elif name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(576, config.NUM_CLASSES)
    elif name == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(1024, config.NUM_CLASSES)
    elif name == 'squeezenet1_0':
        model = models.squeezenet1_0(weights=None)
        model.classifier[1] = nn.Conv2d(512, config.NUM_CLASSES, kernel_size=1)

    return model