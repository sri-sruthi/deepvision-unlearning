import argparse
import torch

from data.cifar_loaders import get_cifar10_loaders, get_cifar100_loaders
from evaluation.metrics import forgetting_effectiveness, retention_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    else:
        train_loader, test_loader = get_cifar100_loaders(args.batch_size)

    model = torch.load(args.model_path)
    fe = forgetting_effectiveness(model, train_loader, device)
    ra = retention_accuracy(model, test_loader, device)

    print("Forgetting Effectiveness:", fe)
    print("Retention Accuracy:", ra)

if __name__ == "__main__":
    main()
