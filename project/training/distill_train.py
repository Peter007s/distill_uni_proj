import torch
import torch.nn as nn
import wandb
import config
from models.embedding import extract_embedding, get_embedding_dim
from training.validation import validate
from distill_loss import distillation_loss
from training.wandb_log import log_embeddings_to_wandb
def train_distillation(teacher, student, train_loader, val_loader,
                       teacher_name='resnet50', student_name='resnet18',
                       epochs=None, lr=None, alpha=0.5, distill_type='cosine'):
    epochs = epochs or config.NUM_EPOCHS
    lr = lr or config.LEARNING_RATE
    
    wandb.init(
        project="distill_uni_proj", 
        name=f"{student_name}_alpha_{alpha}",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": config.BATCH_SIZE,
            "teacher": teacher_name,
            "student": student_name,
            "alpha": alpha,
            "distill_type": distill_type
        },
        reinit=True
    )

    teacher_dim = get_embedding_dim(teacher_name)
    student_dim = get_embedding_dim(student_name)
    projection = nn.Linear(student_dim, teacher_dim).to(config.DEVICE) if teacher_dim != student_dim else None
    params = list(student.parameters()) + (list(projection.parameters()) if projection else [])
    optimizer = torch.optim.Adam(params, lr=lr)
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        student.train()
        train_loss = 0.0
        cls_loss_sum = 0.0
        distill_sum = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            with torch.no_grad():
                teacher_emb = extract_embedding(teacher, inputs)
            
            logits = student(inputs)
            student_emb = extract_embedding(student, inputs, False)
            if projection:
                student_emb = projection(student_emb)
            
            loss, cls_loss, distill_l = distillation_loss(
                logits, labels, student_emb, teacher_emb, alpha, distill_type
            )
            
            cls_loss_sum += cls_loss
            distill_sum += distill_l
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        train_loss /= len(train_loader)
        val_loss, val_acc = validate(student, val_loader)

        avg_cls_loss = cls_loss_sum / len(train_loader)
        avg_distill_loss = distill_sum / len(train_loader)
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "cls_loss": avg_cls_loss,
            "distill_loss": avg_distill_loss
        })

        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Acc: {val_acc:.2f}% | Cls_loss:  {avg_cls_loss:.2f} | distill_loss: {avg_distill_loss:.2f}")

    log_embeddings_to_wandb(
        model=student, 
        dataloader=val_loader, 
        run_name=f"{student_name}_alpha_{alpha}", 
        num_batches=15
    )
    return history