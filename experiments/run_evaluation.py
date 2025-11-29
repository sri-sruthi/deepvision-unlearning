import argparse
import torch

from data.cifar_loaders import get_cifar10_loaders, get_cifar100_loaders
from evaluation.metrics import forgetting_effectiveness, retention_accuracy

# import model architectures
from models.resnet18 import get_resnet18
from models.mobilenetv2 import get_mobilenetv2
from models.vit_tiny import get_vit_tiny
from models.mobilevit import get_mobilevit


def load_model(model_name, num_classes, device, checkpoint_path):
    if model_name == "resnet18":
        model = get_resnet18(num_classes=num_classes)
    elif model_name == "mobilenetv2":
        model = get_mobilenetv2(num_classes=num_classes)
    elif model_name == "vit_tiny":
        model = get_vit_tiny(num_classes=num_classes)
    elif model_name == "mobilevit":
        model = get_mobilevit(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name {model_name}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(args.batch_size)
        num_classes = 10
    else:
        train_loader, test_loader = get_cifar100_loaders(args.batch_size)
        num_classes = 100

    model = load_model(args.model, num_classes, device, args.model_path)
    model.eval()

    fe = forgetting_effectiveness(model, train_loader, device)
    ra = retention_accuracy(model, test_loader, device)

    print("Forgetting Effectiveness:", fe)
    print("Retention Accuracy:", ra)


if __name__ == "__main__":
    main()
