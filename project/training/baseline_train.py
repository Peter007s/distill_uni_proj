import torch
import torch.nn.functional as F
import wandb
import config
from training.validation import validate
from training.wandb_log import log_embeddings_to_wandb
def train_baseline(model, train_loader, val_loader, test_loader, epochs=None, lr=None, model_name="resnet20"):
    epochs = epochs or config.NUM_EPOCHS
    lr = lr or config.LEARNING_RATE
    
    wandb.init(
        project="distill_uni_proj", 
        name=f"baseline_{model_name}",
        mode="offline",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": config.BATCH_SIZE,
            "model": model_name,
            "variant": "baseline"
        },
        reinit=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss, val_acc = validate(model, val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_loss" : val_loss
        })
        
        print(f"Epoch {epoch} | Total Loss: {train_loss:.4f} | Acc: {val_acc:.2f}%")
    
    test_loss, test_acc = validate(model, test_loader)
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})

    log_embeddings_to_wandb(
        model=model,
        dataloader=val_loader, 
        run_name=f"baseline_{model_name}", 
        num_batches=15
    )

    wandb.finish()
    return history