from tqdm import tqdm
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def evaluate_model(model, data_loader, device, criterion = nn.CrossEntropyLoss()):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_size = 0
    resize_transform = transforms.Resize((224, 224), antialias=True)

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            # Resize images here if facing memory issues with whole dataset reshaped at once
            images = torch.stack([resize_transform(img) for img in images])
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_size += labels.size(0)

    average_loss = total_loss / total_size
    accuracy = 100 * total_correct / total_size
    return average_loss, accuracy


def _train(
    model,
    epoch,
    optimizer,
    train_loader,
    criterion=nn.CrossEntropyLoss(),
    log_interval=100,
):
    total_loss = 0
    total_correct = 0
    total_size = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target).sum().item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    total_loss / total_size,
                )
            )

    epoch_loss = total_loss / total_size
    epoch_acc = 100.0 * (total_correct / total_size)
    return epoch_loss, epoch_acc


def train(
    model,
    epochs,
    optimizer,
    train_loader,
    val_loader,
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
):
    best_val_accuracy = 0.0
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        total_size = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_size += data.size(0)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        total_loss / total_size,
                    )
                )

        # Validation phase
        val_accuracy = evaluate_model(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: Validation Accuracy: {val_accuracy}%")

        # Checkpointing
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New best model found at epoch {epoch}. Saving model...")
            torch.save(model.state_dict(), f"best_model_epoch_{epoch}.pth")


def load_model(
    n_classes,
    model_name="efficientnet_v2_s",
    device="cpu",
    pretrained=True,
    transfer_learning=False,
):
    if pretrained:
        model = torch.hub.load(
            "hankyul2/EfficientNetV2-pytorch",
            model_name,
            nclass=n_classes,
            skip_validation=False,
        )
        model.to(device)
    elif transfer_learning:
        #    model = models.efficientnet_b0(pretrained=pretrained)
        model = torch.hub.load(
            "hankyul2/EfficientNetV2-pytorch",
            model_name,
            nclass=n_classes,
            skip_validation=False,
        )
        model.to(device)
    else:
        from tensorflow.keras.applications import EfficientNetB0

        model = EfficientNetB0(weights="imagenet")
    return model


import datetime


def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to memory.
    """
    current_date = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M"
    )  # Format: YYYYMMDD_hhmm

    filename = f"../../checkpoints/efficinet_net_b0_finetune_{current_date}.pth"

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        filename,
    )
