import torch.nn as nn
import timm

def get_vit_tiny(num_classes=10, pretrained=False):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
