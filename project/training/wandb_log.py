import torch
import wandb
import config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models.embedding import extract_embedding
# training/wandb_log.py
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models.embedding import extract_embedding
import config

def log_pair_embeddings(model_a, model_b, dataloader, name_a, name_b, run_name, num_batches=15, method='PCA'):
    model_a.eval()
    model_b.eval()
    
    embs_a, embs_b, all_labels = [], [], []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= num_batches: break
            inputs = inputs.to(config.DEVICE, non_blocking=True)
            embs_a.extend(extract_embedding(model_a, inputs).cpu().numpy())
            embs_b.extend(extract_embedding(model_b, inputs).cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_labels = np.array(all_labels)
    combined = np.vstack([embs_a, embs_b])
    
    reducer = PCA(n_components=2) if method == 'PCA' else TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = reducer.fit_transform(combined)
    
    n = len(embs_a)
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap('tab10', len(config.CIFAR10_CLASSES))
    
    for i, cls_name in enumerate(config.CIFAR10_CLASSES):
        mask = all_labels == i
        ax.scatter(reduced[:n, 0][mask], reduced[:n, 1][mask], 
                   c=[cmap(i)], marker='o', s=25, alpha=0.7, edgecolor='none',
                   label=f'{name_a}' if i == 0 else None)
        ax.scatter(reduced[n:, 0][mask], reduced[n:, 1][mask], 
                   c=[cmap(i)], marker='x', s=35, alpha=0.9,
                   label=f'{name_b}' if i == 0 else None)
        
    ax.set_title(f"{method}: {name_a} vs {name_b} ({run_name})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(title="Models", loc='best')
    
    wandb.log({f"embeddings/{method}_{name_a}_vs_{name_b}": wandb.Image(fig)})
    plt.close(fig)

def log_embeddings_to_wandb(model, dataloader, run_name, num_batches=20, method='PCA'):
    print(f"Generating {method} for {run_name}")
    
    all_embs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(config.DEVICE, non_blocking=True)
            embs = extract_embedding(model, inputs)
            
            all_embs.extend(embs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    if method == 'PCA':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    
    reduced_embs = reducer.fit_transform(all_embs)
    class_names = config.CIFAR10_CLASSES
    
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(all_labels)
    cmap = plt.get_cmap('tab10', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(all_labels) == label
        ax.scatter(reduced_embs[mask, 0], reduced_embs[mask, 1], 
                   c=[cmap(i)], label=class_names[int(label)], s=15, alpha=0.7, edgecolor='w')
        
    ax.set_title(f"{method} Embeddings: {run_name}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title="Class", loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    wandb.log({f"embeddings/{method}_{run_name}": wandb.Image(fig)})
    plt.close(fig)  # Prevents memory leaks
    
    print(f"embeddings/{method}_{run_name}")