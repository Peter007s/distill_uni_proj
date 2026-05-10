from torch.utils.data import DataLoader
from torchvision import transforms
import config
import data.load_dataset
def create_dataloaders(train_ds, val_ds, test_ds):
    
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
    
    train_ds.transform = train_tf
    val_ds.transform = eval_tf
    test_ds.transform = eval_tf
    
    return (
        DataLoader(train_ds, config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS),
        DataLoader(val_ds, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS),
        DataLoader(test_ds, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    )
if __name__ == "__main__":
    train_ds, val_ds, test_ds = data.load_dataset.load_dataset()
    create_dataloaders(train_ds, val_ds, test_ds)