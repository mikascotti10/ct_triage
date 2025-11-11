import torch
import torch.nn as nn
from torchvision import models

def make_resnet18(num_classes=2, pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
