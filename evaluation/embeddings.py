import torch
import numpy as np

def extract_layer_embeddings(model, dataloader, layer, device):
    embeddings = []

    def hook(module, input, output):
        embeddings.append(output.detach().cpu())

    handle = layer.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            model(images)

    handle.remove()
    return torch.cat(embeddings, dim=0)

def cosine_distance(a, b):
    return 1 - torch.nn.functional.cosine_similarity(a, b).mean().item()
