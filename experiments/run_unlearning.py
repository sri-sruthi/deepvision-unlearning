import argparse
import torch
import os

from data.cifar_loaders import get_cifar10_loaders, get_cifar100_loaders
from forget_sets.selector import get_class_forget_loader, get_sample_forget_loader
from unlearning_methods.gradient_ascent import gradient_ascent_unlearn
from unlearning_methods.sisa_unlearning import sisa_retrain
from utils.trainer import validate
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
        raise ValueError("Unknown model")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="ga", choices=["ga", "sisa"])
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--forget-type", type=str, default="class", choices=["class", "sample"])
    parser.add_argument("--forget-class", type=str, default="airplane")
    parser.add_argument("--forget-count", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="./models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset selection
    if args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(args.batch_size)
        num_classes = 10
        class_map = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
                     "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
    else:
        train_loader, test_loader = get_cifar100_loaders(args.batch_size)
        num_classes = 100
        class_map = None

    forget_loader = None

    # Forget-set selection
    if args.forget_type == "class":
        forget_loader, forget_indices = get_class_forget_loader(train_loader.dataset, class_map[args.forget_class])
    else:
        forget_loader, forget_indices = get_sample_forget_loader(train_loader.dataset, count=args.forget_count)

    # Load pretrained baseline
    model_path = f"{args.save_dir}/{args.model}_{args.dataset}_best.pth"
    model = build_model(args.model, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))

    # Perform unlearning
    if args.method == "ga":
        model = gradient_ascent_unlearn(model, forget_loader, device=device)
    else:
        model = sisa_retrain(model, forget_loader, device=device)

    # Save unlearned model
    unlearn_name = f"{args.model}_{args.dataset}_{args.method}_{args.forget_type}.pth"
    torch.save(model.state_dict(), f"{args.save_dir}/{unlearn_name}")
    print(f"Saved unlearned model: {unlearn_name}")


if __name__ == "__main__":
    main()
