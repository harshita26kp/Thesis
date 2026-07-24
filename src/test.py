import torch

from model import create_model
from dataset import get_dataloaders

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model = create_model().to(device)

model.eval()

def test_model(model, dataloader, device):

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


if __name__ == "__main__":

    try:

        _, test_loader = get_dataloaders()

        accuracy = test_model(
            model,
            test_loader,
            device
        )

        print(f"Test Accuracy: {accuracy:.2f}%")

    except Exception:

        print("Dataset not found.")
        print("Testing will be available once the dataset is added.")