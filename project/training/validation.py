import torch
import config
import torch.nn.functional as F

def validate(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            correct += logits.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)
            
    val_loss = total_loss / len(dataloader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc