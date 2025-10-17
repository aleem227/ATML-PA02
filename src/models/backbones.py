# src/models/backbones.py
import torch.nn as nn
import torchvision.models as tv

def build_backbone(name="resnet50", pretrained=True, feat_dim=2048, freeze=False):
    if name.lower() == "resnet50":
        m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        backbone = m
    else:
        raise ValueError("Only resnet50 implemented in this minimal starter.")

    if freeze:
        for p in backbone.parameters(): p.requires_grad = False
    return backbone, feat_dim

class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
