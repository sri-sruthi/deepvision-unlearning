import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os

from data.cifar_loaders import get_cifar10_loaders, get_cifar100_loaders
from utils.trainer import train_one_epoch, validate
from models.resnet18 import get_resnet18
from models.mobilenetv2 import get_mobilenetv2
from models.vit_tiny import get_vit_tiny
from models.mobilevit import get_mobilevit


def build_model(model_name, num_classes):
    if model_name == "resnet18":
        return get_resnet18(num_classes)
    elif model_name == "mobilenetv2":
        return get_mobilenetv2(num_classes)
    elif model_name == "vit_tiny":
        return get_vit_tiny(num_classes)
    elif model_name == "mobilevit":
        return get_mobilevit(num_classes)
    else:
        raise ValueError("Unknown model name")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_dir", type=str, default="./models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(args.batch_size)
        num_classes = 10
    else:
        train_loader, test_loader = get_cifar100_loaders(args.batch_size)
        num_classes = 100

    model = build_model(args.model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.save_dir}/{args.model}_{args.dataset}_best.pth")
            print("Saved Best Model")

    print("Training complete! Best Acc:", best_acc)


if __name__ == "__main__":
    main()
