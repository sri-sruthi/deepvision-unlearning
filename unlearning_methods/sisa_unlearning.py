import torch
import torch.nn as nn
from tqdm import tqdm
import copy

def sisa_retrain(model, retrain_loader, epochs=5, lr=0.001, device="cuda"):
    new_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in tqdm(retrain_loader, desc=f"SISA Retraining (Epoch {epoch+1})"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(new_model(images), labels)
            loss.backward()
            optimizer.step()

    return new_model
