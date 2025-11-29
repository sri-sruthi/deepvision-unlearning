import torch
import torch.nn as nn
from tqdm import tqdm

def gradient_ascent_unlearn(model, forget_loader, lr=0.001, steps=50, device="cuda"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in tqdm(range(steps), desc="Unlearning via Gradient Ascent"):
        for images, labels in forget_loader:

            images = images.to(device)

            # Ensure labels are tensors
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)

            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = -criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model
