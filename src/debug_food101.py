from pathlib import Path
from torchvision.datasets import Food101

DATASET_PATH = Path(__file__).resolve().parent.parent / "data"

print("Creating test dataset...")

test_dataset = Food101(
    root=DATASET_PATH,
    split="test",
    download=False
)

print("Dataset created.")
print("Length:", len(test_dataset))

print("Loading first image...")
image, label = test_dataset[0]

print("First image loaded successfully.")
print("Label:", label)