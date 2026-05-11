import torch
import config

def validate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total