import random
from torch.utils.data import Subset, DataLoader

def get_class_forget_loader(dataset, forget_class, batch_size=64):
    forget_indices = [i for i, (_, label) in enumerate(dataset) if label == forget_class]
    subset = Subset(dataset, forget_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True), forget_indices


def get_sample_forget_loader(dataset, count=500, batch_size=64):
    total_indices = list(range(len(dataset)))
    forget_indices = random.sample(total_indices, count)
    subset = Subset(dataset, forget_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True), forget_indices


def get_subclass_forget_loader(dataset, coarse_to_fine_map, forget_coarse_class, batch_size=64):
    fine_classes = coarse_to_fine_map[forget_coarse_class]
    forget_indices = [i for i, (_, label) in enumerate(dataset) if label in fine_classes]
    subset = Subset(dataset, forget_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True), forget_indices
