import torch
import torch.nn.functional as F
import config
from training.validation import validate
def train_baseline(model, train_loader, val_loader, epochs=None, lr=None):
    epochs = epochs or config.NUM_EPOCHS
    lr = lr or config.LEARNING_RATE
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_acc = validate(model, val_loader)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        print('epoch done')
    
    return history