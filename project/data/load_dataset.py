from pathlib import Path
from tinyimagenet import TinyImageNet
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import config
def load_dataset():

    dir = Path(__file__).parent
    dir = dir / "tinyimagenet"

    full_train = TinyImageNet(dir, split='train')
    
    train_idx, val_idx = train_test_split(
        range(len(full_train)),
        test_size=config.VAL_SIZE,
        stratify=full_train.targets,
    )
    
    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_train, val_idx)
    test_ds = TinyImageNet(dir, split='val')
    
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    path = load_dataset()
    print('gg', path)