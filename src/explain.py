import torch

from model import create_model

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    AblationCAM
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model = create_model().to(device)


model.eval()

target_layers = [model.layer4[-1]]

grad_cam = GradCAM(
    model=model,
    target_layers=target_layers
)

grad_cam_pp = GradCAMPlusPlus(
    model=model,
    target_layers=target_layers
)

score_cam = ScoreCAM(
    model=model,
    target_layers=target_layers
)

ablation_cam = AblationCAM(
    model=model,
    target_layers=target_layers
)

print("Explainability methods initialized successfully.")