import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def compute_auc(scores):

    x = np.linspace(0, 1, len(scores))

    auc = np.trapezoid(scores, x)

    return float(auc)


def predict_probability(model, image_tensor, class_index):

    model.eval()

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

    return probabilities[0, class_index].item()


def prepare_heatmap(cam_mask, image_size=(224, 224)):

    heatmap = cv2.resize(cam_mask, image_size)

    heatmap = heatmap.astype(np.float32)

    heatmap -= heatmap.min()

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    flat_heatmap = heatmap.flatten()

    sorted_indices = np.argsort(-flat_heatmap)

    return heatmap, sorted_indices


def deletion_metric(
    model,
    image_tensor,
    cam_mask,
    class_index,
    steps=100
):

    _, sorted_indices = prepare_heatmap(cam_mask)

    deleted_image = image_tensor.clone()

    flat_image = deleted_image.view(3, -1)

    total_pixels = flat_image.shape[1]

    pixels_per_step = max(
        1,
        total_pixels // steps
    )

    scores = []

    for step in range(steps + 1):

        score = predict_probability(
            model,
            deleted_image,
            class_index
        )

        scores.append(score)

        if step == steps:
            break

        start = step * pixels_per_step

        end = min(
            (step + 1) * pixels_per_step,
            total_pixels
        )

        indices = sorted_indices[start:end]

        flat_image[:, indices] = 0

    auc = compute_auc(scores)

    return scores, auc

def insertion_metric(
    model,
    image_tensor,
    cam_mask,
    class_index,
    steps=100
):


    _, sorted_indices = prepare_heatmap(cam_mask)

    original_image = image_tensor.clone()

    inserted_image = torch.zeros_like(image_tensor)

    flat_original = original_image.view(3, -1)
    flat_inserted = inserted_image.view(3, -1)

    total_pixels = flat_original.shape[1]

    pixels_per_step = max(
        1,
        total_pixels // steps
    )

    scores = []

    for step in range(steps + 1):

        score = predict_probability(
            model,
            inserted_image,
            class_index
        )

        scores.append(score)

        if step == steps:
            break

        start = step * pixels_per_step

        end = min(
            (step + 1) * pixels_per_step,
            total_pixels
        )

        indices = sorted_indices[start:end]

        # Restore the important pixels
        flat_inserted[:, indices] = flat_original[:, indices]

    auc = compute_auc(scores)

    return scores, auc


def plot_metric(scores, title):

    plt.figure(figsize=(6, 4))

    x = np.linspace(0, 100, len(scores))

    plt.plot(x, scores)

    plt.xlabel("Pixels Removed (%)")

    plt.ylabel("Prediction Probability")

    plt.title(title)

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=300)
    plt.close()