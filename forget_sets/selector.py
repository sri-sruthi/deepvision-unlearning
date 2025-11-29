import random
from torch.utils.data import Subset

def get_class_forget_loader(dataset, forget_class, batch_size=64):
    forget_indices = [i for i, (_, label) in enumerate(dataset) if label == forget_class]
    return Subset(dataset, forget_indices), forget_indices


def get_sample_forget_loader(dataset, count=500, batch_size=64):
    total_indices = list(range(len(dataset)))
    forget_indices = random.sample(total_indices, count)
    return Subset(dataset, forget_indices), forget_indices


def get_subclass_forget_loader(dataset, coarse_to_fine_map, forget_coarse_class, batch_size=64):
    fine_classes = coarse_to_fine_map[forget_coarse_class]
    forget_indices = [i for i, (_, label) in enumerate(dataset) if label in fine_classes]
    return Subset(dataset, forget_indices), forget_indices
