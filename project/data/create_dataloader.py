import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import config

train_tf = transforms.Compose([
    transforms.RandomCrop(config.IMG_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_and_create_dataloaders():
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
    
    val_size = int(len(full_train_dataset) * config.VAL_SIZE)
    train_size = len(full_train_dataset) - val_size
    
    train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])
    
    val_ds.dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=eval_tf)
    
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_tf)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,
    pin_memory=True,
        persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS,
    pin_memory=True,
        persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS,
    pin_memory=True, 
        persistent_workers=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_load, val_load, test_load = load_and_create_dataloaders()