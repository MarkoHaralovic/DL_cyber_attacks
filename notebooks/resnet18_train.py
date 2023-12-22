"""
Train CIFAR10 with PyTorch using Resnet18 architecture model.

Run from the base of the project.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm

import os
import sys
from datetime import datetime
import time

from Data import Data
from auxiliary import format_time

sys.path.append("models")
from resnet18 import ResNet18

# Params
LR = 0.1
WD = 5e-4
MOMENTUM = 0.9
NUM_EPOCHS = 50

BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

EXP_NAME = "test"
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")


# Training
def train(epoch):
    start_time = time.time()
    
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
    
    end_time = time.time()  
    training_time = end_time - start_time 

    print(f"Training Time for Epoch {epoch}: {training_time} seconds")
    return train_loss, 100.0 * correct / total, training_time


def test(epoch):
    start_time = time.time()
    
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
    checkpoints_dir = os.path.join(os.path.curdir, "checkpoints")
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }

        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)

        torch.save(
            state,
            os.path.join(checkpoints_dir, f"resnet18_ckpt_{TIMESTAMP}.pth"),
        )
        best_acc = acc
    
    end_time = time.time()  
    testing_time = end_time - start_time
    print(f"Testing Time for Epoch {epoch}: {testing_time} seconds")
    
    return test_loss, 100.0 * correct / total, testing_time


def plot_metrics(train_loss, train_acc, val_loss, val_acc):
    epochs = np.arange(NUM_EPOCHS)

    if max(val_loss) - min(train_loss) < 50:
        fig, (loss_ax, acc_ax) = plt.subplots(
            1, 2, figsize=(12, 5), layout="constrained"
        )
        loss_ax.plot(epochs, train_loss, label="train")
        loss_ax.plot(epochs, val_loss, label="val")
        loss_ax.set(title="Loss")

        axes = [loss_ax, acc_ax]
    else:
        # Plot train and validation loss separately
        fig, ((train_loss_ax, acc_ax), (val_loss_ax, empty)) = plt.subplots(
            2, 2, figsize=(12, 10), layout="constrained"
        )
        empty.set(visible=False)

        train_loss_ax.plot(epochs, train_loss)
        val_loss_ax.plot(epochs, val_loss)
        train_loss_ax.set(title="Train loss")
        val_loss_ax.set(title="Val loss")

        axes = [train_loss_ax, acc_ax, val_loss_ax]

    acc_ax.plot(epochs, train_acc, label="train")
    acc_ax.plot(epochs, val_acc, label="val")
    acc_ax.set_title("Accuracy")
    acc_ax.legend()

    for ax in axes:
        ax.grid(True, linestyle=":")
        ax.set_xlabel("Epoch")

    fig.suptitle(
        f"Resnet18 - {EXP_NAME}\n[LR={LR}, WD={WD}, M={MOMENTUM}, BS={BATCH_SIZE}]",
        fontsize=14,
    )
    plt.savefig(
        os.path.join(os.path.curdir, "checkpoints", f"resnet18_metrics_{TIMESTAMP}.svg")
    )


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

    # Collect metrics
    train_loss_data = np.zeros(NUM_EPOCHS)
    train_acc_data = np.zeros(NUM_EPOCHS)
    val_loss_data = np.zeros(NUM_EPOCHS)
    val_acc_data = np.zeros(NUM_EPOCHS)

    total_training_time = 0
    total_testing_time = 0  
    
    # Train
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, training_time = train(epoch)
        train_loss_data[epoch] = train_loss
        train_acc_data[epoch] = train_acc
        total_training_time += training_time

        val_loss, val_acc, testing_time = test(epoch)
        val_loss_data[epoch] = val_loss
        val_acc_data[epoch] = val_acc
        total_testing_time += testing_time

        scheduler.step()

    print(f"Total Training Time: {format_time(total_training_time)}")
    print(f"Total Testing Time: {format_time(total_testing_time)}") 
    
    plot_metrics(train_loss_data, train_acc_data, val_loss_data, val_acc_data)
