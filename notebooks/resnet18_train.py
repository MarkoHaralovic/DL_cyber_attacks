"""
Train CIFAR10 with PyTorch.

Run from the base of the project.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import sys
from datetime import datetime
from tqdm import tqdm

from Data import Data

sys.path.append("models")
from resnet18 import ResNet18

# Params
LR = 0.1
WD = 5e-4
MOMENTUM = 0.9
NUM_EPOCHS = 50

BATCH_SIZE = 256
NUM_WORKERS = 0

EXP_NAME = "test"
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(
        enumerate(trainloader), desc=f"Epoch {epoch}", total=len(trainloader)
    ):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(
        "Loss: %.3f | Acc: %.3f%% (%d/%d)"
        % (train_loss, 100.0 * correct / total, correct, total)
    )


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(
            enumerate(testloader), f"Validating epoch {epoch}", total=len(testloader)
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(
        "Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)"
        % (test_loss, 100.0 * correct / total, correct, total)
    )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(
            state,
            os.path.join(
                os.path.curdir, "checkpoints", f"resnet18_ckpt_{TIMESTAMP}.pth"
            ),
        )
        best_acc = acc


if __name__ == "__main__":
    datasets_folder = os.path.join(os.path.curdir, "datasets", "CIFAR10", "cifar-10")
    train_images = os.path.join(datasets_folder, "train", "data.npy")
    train_labels = os.path.join(datasets_folder, "train", "labels.npy")
    test_images = os.path.join(datasets_folder, "test", "data.npy")
    test_labels = os.path.join(datasets_folder, "test", "labels.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    best_acc = 0  # best test accuracy
    # start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Model
    print("==> Building model..")
    net = ResNet18()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    cifar_10_dataset = Data(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        train_transform=transform_train,
        test_transform=transform_test,
    )

    testset = TensorDataset(
        torch.tensor(cifar_10_dataset.test_images, dtype=torch.float32).permute(
            0, 3, 1, 2
        ),
        torch.tensor(cifar_10_dataset.test_labels, dtype=torch.long),
    )
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    trainset = TensorDataset(
        torch.tensor(cifar_10_dataset.train_images, dtype=torch.float32).permute(
            0, 3, 1, 2
        ),
        torch.tensor(cifar_10_dataset.train_labels, dtype=torch.long),
    )
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    classes = cifar_10_dataset.classes

    for epoch in range(NUM_EPOCHS):
        train(epoch)
        test(epoch)
        scheduler.step()
