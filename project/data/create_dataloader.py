from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import config
train_tf = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_tf = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def apply_train_transforms(examples):
    examples['image'] = [train_tf(image.convert("RGB")) for image in examples['image']]
    return examples

def apply_eval_transforms(examples):
    examples['image'] = [eval_tf(image.convert("RGB")) for image in examples['image']]
    return examples


def tuple_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return images, labels
# --- 2. YOUR CLEANER LOAD FUNCTION ---

def load_and_create_dataloaders():
    dataset = load_dataset('zh-plus/tiny-imagenet')

    # Re-using your split logic
    test_data = dataset['valid']
    train_val_split = dataset['train'].train_test_split(
        test_size=config.VAL_SIZE,
        stratify_by_column="label"
    )
    
    train_data = train_val_split['train']
    val_data = train_val_split['test'] 


    train_ds = train_data.with_transform(apply_train_transforms)
    val_ds = val_data.with_transform(apply_eval_transforms) 
    test_ds = test_data.with_transform(apply_eval_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=tuple_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=tuple_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=tuple_collate_fn)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_load, val_load, test_load = load_and_create_dataloaders()