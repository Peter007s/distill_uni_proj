import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve()))

import torch
import config
from models.embedding import *
from models.student_model import *
from models.teacher_model import *
from training.baseline_train import *
from training.distill_train import *
from distill_loss import *
from data.create_dataloader import *
if __name__ == '__main__':
    train_load, val_load, test_load = load_and_create_dataloaders()
    
    hist = train_distillation(
        teacher=teacher_model(),
        student=create_student('resnet20'),
        train_loader=train_load,
        val_loader=val_load,
        teacher_name='cifar10_resnet56',
        student_name='resnet20',
        epochs=10
    )