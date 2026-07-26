import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from pathlib import Path
from PIL import Image

from torchvision import transforms
from torchvision.datasets import Food101

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    AblationCAM,
)

from pytorch_grad_cam.utils.image import show_cam_on_image

from model import create_model
from metrics import deletion_metric, insertion_metric


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():

    model = create_model()

    model.load_state_dict(
        torch.load(
            "outputs/models/resnet50_food101.pth",
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    return model


def generate_cam(model, image, input_tensor, cam_class):

    target_layers = [model.layer4[-1]]

    cam = cam_class(
    model=model,
    target_layers=target_layers,
)

    grayscale_cam = cam(
    input_tensor=input_tensor
    )[0]

    rgb_image = (
        np.array(image.resize((224, 224)))
        .astype(np.float32)
        / 255.0
    )

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
    )

    return visualization, grayscale_cam


def predict(model, input_tensor):

    with torch.no_grad():

        output = model(input_tensor)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

    return prediction

def main():

    print("Loading model...")
    model = load_model()

    print("Loading Food-101 test dataset...")

    test_dataset = Food101(
        root="data",
        split="test",
        transform=transform,
        download=False,
    )

    methods = {
        "Grad-CAM": GradCAM,
        "Grad-CAM++": GradCAMPlusPlus,
        "Score-CAM": ScoreCAM,
        "Ablation-CAM": AblationCAM,
    }

    results = []

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    num_images = 100

    indices = random.sample(
    range(len(test_dataset)),
    num_images
)

    for image_number, index in enumerate(indices, start=1):

        print(f"\nProcessing image {image_number}/{num_images}")

        image, label = test_dataset[index]

        input_tensor = image.unsqueeze(0).to(device)

        prediction = predict(
            model,
            input_tensor
        )

        pil_image = Image.open(
            test_dataset._image_files[index]
        ).convert("RGB")

        class_name = test_dataset.classes[label]

        for method_name, cam_class in methods.items():

            print(f"  {method_name}")

            visualization, mask = generate_cam(
                model,
                pil_image,
                input_tensor,
                cam_class,
            )

            deletion_scores, deletion_auc = deletion_metric(
                model,
                input_tensor,
                mask,
                prediction,
            )

            insertion_scores, insertion_auc = insertion_metric(
                model,
                input_tensor,
                mask,
                prediction,
            )

            results.append({
                "Image": index,
                "Class": class_name,
                "Method": method_name,
                "Deletion AUC": deletion_auc,
                "Insertion AUC": insertion_auc,
            })

    results_df = pd.DataFrame(results)

    csv_path = output_dir / "xai_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\nResults saved to: {csv_path}")

    mean_results = (
        results_df
        .groupby("Method")[["Deletion AUC", "Insertion AUC"]]
        .mean()
        .reset_index()
    )

    mean_csv = output_dir / "mean_results.csv"
    mean_results.to_csv(mean_csv, index=False)

    print(f"Mean results saved to: {mean_csv}")

    plt.figure(figsize=(8, 5))

    plt.bar(
        mean_results["Method"],
        mean_results["Deletion AUC"]
    )

    plt.ylabel("Mean Deletion AUC")
    plt.title("Deletion Metric Comparison")

    plt.tight_layout()

    plt.savefig(
        output_dir / "deletion_bar_chart.png",
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.bar(
        mean_results["Method"],
        mean_results["Insertion AUC"]
    )

    plt.ylabel("Mean Insertion AUC")
    plt.title("Insertion Metric Comparison")

    plt.tight_layout()

    plt.savefig(
        output_dir / "insertion_bar_chart.png",
        dpi=300
    )

    plt.close()

    print("\n========== Average Results ==========\n")

    print(mean_results)


if __name__ == "__main__":
    main()