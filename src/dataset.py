from pathlib import Path

from torchvision.datasets import Food101
from torchvision import transforms
from torch.utils.data import DataLoader

DATASET_PATH = Path(__file__).resolve().parent.parent / "data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_dataloaders(batch_size=16):

    print("Loading training dataset...")

    train_dataset = Food101(
        root=DATASET_PATH,
        split="train",
        transform=transform,
        download=False
    )

    print("Training dataset loaded.")

    print("Loading test dataset...")

    test_dataset = Food101(
        root=DATASET_PATH,
        split="test",
        transform=transform,
        download=False
    )

    print("Test dataset loaded.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    print("DataLoaders created.")

    return train_loader, test_loader