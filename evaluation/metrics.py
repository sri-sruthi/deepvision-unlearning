import torch
import numpy as np

def forgetting_effectiveness(model, forget_loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in forget_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    forgetting_acc = correct / total
    return forgetting_acc  # Lower is better (closer to random guessing)
    

def retention_accuracy(model, retained_loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in retained_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total
