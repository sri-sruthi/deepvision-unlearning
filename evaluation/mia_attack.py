import torch

def simple_mia_attack(model, loader, device, threshold=0.5):
    model.eval()
    private_scores = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            top_probs, _ = probs.max(dim=1)
            private_scores.extend(top_probs.cpu().numpy())

    return sum(score > threshold for score in private_scores) / len(private_scores)
