import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_embeddings(embeddings, labels, title="t-SNE Visualization", save_path=None):
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    reduced = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:,0], reduced[:,1], c=labels_np, cmap="tab10", alpha=0.7)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
