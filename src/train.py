import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from model import create_model
from dataset import get_dataloaders

MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = create_model().to(device)


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):

        print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy



def save_model(model, filename="resnet50_food101.pth"):

    save_path = MODEL_DIR / filename

    torch.save(model.state_dict(), save_path)

    print(f"Model saved to: {save_path}")


if __name__ == "__main__":

    try:

        train_loader, _ = get_dataloaders()

        EPOCHS = 1

        for epoch in range(EPOCHS):

            loss, acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            print(
                f"Epoch {epoch+1}/{EPOCHS}"
                f" | Loss: {loss:.4f}"
                f" | Accuracy: {acc:.2f}%"
            )

        
        save_model(model)

    except Exception as e:
     print(e)