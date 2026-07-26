import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from torchvision import transforms
from torchvision.datasets import Food101
from metrics import deletion_metric, insertion_metric

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    AblationCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import create_model


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def load_model():

    model = create_model()

    model.load_state_dict(
        torch.load(
            "outputs/models/resnet50_food101.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


def load_classes():

    dataset = Food101(
        root="data",
        split="train",
        download=False
    )

    return dataset.classes


def load_image(image_path):

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    return image, input_tensor

def predict(model, input_tensor, classes):

    with torch.no_grad():
        outputs = model(input_tensor)

    predicted_index = outputs.argmax(dim=1).item()
    predicted_class = classes[predicted_index]

    return predicted_index, predicted_class


def generate_cam(model, image, input_tensor, cam_class):

    target_layers = [model.layer4[-1]]

    cam = cam_class(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_image = np.array(
        image.resize((224, 224))
    ).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization, grayscale_cam

def save_visualization(image, filename):

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / filename

    plt.imsave(save_path, image)

    print(f"Saved to: {save_path}")

def main():

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")

    model = load_model()
    print("Model loaded successfully.")

    classes = load_classes()

    image_path = Path(
        "data/food-101/images/bibimbap/2556.jpg"
    )

    actual_class = image_path.parent.name

    image, input_tensor = load_image(image_path)

    predicted_index, predicted_class = predict(
        model,
        input_tensor,
        classes
    )

    print(f"Image: {image_path.name}")
    print(f"Actual class: {actual_class}")
    print(f"Predicted class index: {predicted_index}")
    print(f"Predicted class: {predicted_class}")

    methods = {
        "Grad-CAM": GradCAM,
        "Grad-CAM++": GradCAMPlusPlus,
        "Score-CAM": ScoreCAM,
        "Ablation-CAM": AblationCAM,
    }

    results = {}


    for name, cam_class in methods.items():

        print(f"\nGenerating {name}...")

        visualization, mask = generate_cam(
            model,
            image,
            input_tensor,
            cam_class
        )

        filename = (
            name.lower()
            .replace("+", "plus")
            .replace("-", "")
            .replace(" ", "_")
            + "_bibimbap_2556.png"
        )

        save_visualization(
            visualization,
            filename
        )

        deletion_scores, deletion_auc = deletion_metric(
            model,
            input_tensor,
            mask,
            predicted_index
        )

        insertion_scores, insertion_auc = insertion_metric(
            model,
            input_tensor,
            mask,
            predicted_index
        )

        results[name] = {
            "image": visualization,
            "mask": mask,
            "deletion_scores": deletion_scores,
            "deletion_auc": deletion_auc,
            "insertion_scores": insertion_scores,
            "insertion_auc": insertion_auc
        }

        print(f"Deletion AUC : {deletion_auc:.4f}")
        print(f"Insertion AUC: {insertion_auc:.4f}")

    plt.figure(figsize=(20, 5))

    for i, (name, data) in enumerate(results.items(), start=1):

        plt.subplot(1, len(results), i)
        plt.imshow(data["image"])
        plt.title(name)
        plt.axis("off")

        plt.tight_layout()

    comparison_path = Path("outputs/figures/cam_comparison.png")
    plt.savefig(comparison_path, dpi=300)

    print(f"\nComparison figure saved to: {comparison_path}")

    plt.close()

    print("\n========== XAI Evaluation Summary ==========\n")

    all_results = []

    for name, data in results.items():
        print(
            f"{name:<15}"
            f"Deletion AUC = {data['deletion_auc']:.4f}    "
            f"Insertion AUC = {data['insertion_auc']:.4f}"
        )


if __name__ == "__main__":
    main()  