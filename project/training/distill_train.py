import torch
import torch.nn as nn
import config
from models.embedding import extract_embedding, get_embedding_dim
from training.validation import validate
from distill_loss import distillation_loss

def train_distillation(teacher, student, train_loader, val_loader,
                       teacher_name='resnet50', student_name='resnet18',
                       epochs=None, lr=None, alpha=0.5, distill_type='mse'):
    epochs = epochs or config.NUM_EPOCHS
    lr = lr or config.LEARNING_RATE
    
    
    teacher_dim = get_embedding_dim(teacher_name)
    student_dim = get_embedding_dim(student_name)
    projection = nn.Linear(student_dim, teacher_dim).to(config.DEVICE) if teacher_dim != student_dim else None
    params = list(student.parameters()) + (list(projection.parameters()) if projection else [])
    optimizer = torch.optim.Adam(params, lr=lr)
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        student.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            with torch.no_grad():
                teacher_emb = extract_embedding(teacher, inputs)
            
            logits = student(inputs)
            student_emb = extract_embedding(student, inputs)
            if projection:
                student_emb = projection(student_emb)
            
            loss, cls_loss, distill_l = distillation_loss(
                logits, labels, student_emb, teacher_emb, alpha, distill_type
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_acc = validate(student, val_loader)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        print(epoch, train_loss, val_acc)
    
    return history