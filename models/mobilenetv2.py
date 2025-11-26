import torch.nn as nn
import torchvision.models as models

def get_mobilenetv2(num_classes=10, pretrained=False):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
