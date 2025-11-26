import torch.nn as nn
import timm

def get_mobilevit(num_classes=10, pretrained=False):
    model = timm.create_model("mobilevit_s", pretrained=pretrained)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
