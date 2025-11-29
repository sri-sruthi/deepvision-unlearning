import torch
import torch.nn as nn
from tqdm import tqdm

def gradient_ascent_unlearn(model, forget_loader, lr=0.001, steps=50, device="cuda"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in tqdm(range(steps), desc="Unlearning via Gradient Ascent"):
        for batch in forget_loader:

            # Support loaders returning (images, labels) pairs
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, labels = batch
            else:
                raise ValueError("Forget loader must return (images, labels) batch format")

            # Move to device
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # NEGATE LOSS FOR GRADIENT ASCENT
            loss = -criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model
