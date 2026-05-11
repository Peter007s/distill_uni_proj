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
from training.wandb_log import *
if __name__ == '__main__':
    train_load, val_load, test_load = load_and_create_dataloaders()
    wandb.init(
        project="distill_uni_proj", 
        name="gen_graph",
        config={
        },
        reinit=True
    )
    log_embeddings_to_wandb(teacher_model(), val_load, 'gen_graph')
