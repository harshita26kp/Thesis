import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_model(num_classes=101):
   
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":

    model = create_model()

    print(model)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nTotal Parameters: {total_params:,}")