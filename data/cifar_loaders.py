import torchvision
from torch.utils.data import DataLoader
from utils.transforms import get_cifar10_transforms, get_cifar100_transforms

def get_cifar10_loaders(batch_size=128, num_workers=2, data_root='./data'):
    transform_train, transform_test = get_cifar10_transforms()

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_cifar100_loaders(batch_size=128, num_workers=2, data_root='./data'):
    transform_train, transform_test = get_cifar100_transforms()

    train_set = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
