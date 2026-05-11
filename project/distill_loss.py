import torch.nn.functional as F
import config

def distillation_loss(logits=None, labels=None, 
                      student_emb=None, teacher_emb=None,
                      alpha=config.DISTILL_CLS_LOSS, distill_type='mse'):
    
    if distill_type == 'mse':
        distill_loss = F.mse_loss(student_emb, teacher_emb)
    elif distill_type == 'cosine':
        distill_loss = (1 - F.cosine_similarity(student_emb, teacher_emb, dim=1)).mean()

    cls_loss = F.cross_entropy(logits, labels)
    total_loss = alpha * distill_loss + (1 - alpha) * cls_loss
    
    return total_loss, cls_loss.item(), distill_loss.item()