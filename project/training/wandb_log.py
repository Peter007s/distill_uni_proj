import torch
import wandb
import config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models.embedding import extract_embedding

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