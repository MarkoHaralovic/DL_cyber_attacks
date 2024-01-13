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
sys.path.append("models_functions")
from resnet_functions import train,test,plot_metrics

# Params
LR = 0.001
WD = 5e-4
MOMENTUM = 0.9
NUM_EPOCHS = 300

BATCH_SIZE = 128 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)
EXP_NAME = "badnets_grid"
POISONED_RATE = "5_percent"
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")

CLEAN = True 

if __name__ == "__main__":
    if CLEAN:
        # Clean data
        print(f" Current directory is {os.path.curdir}")
        datasets_folder = os.path.join(os.path.curdir, "datasets", "CIFAR10", "cifar-10")
        train_images = os.path.join(datasets_folder, "train", "data.npy")
        train_labels = os.path.join(datasets_folder, "train", "labels.npy")
        test_images = os.path.join(datasets_folder, "test", "data.npy")
        test_labels = os.path.join(datasets_folder, "test", "labels.npy")
    else:
        print(f" Current directory is {os.path.curdir}")
        # Poisoned data
        datasets_folder = os.path.join(os.path.curdir, "datasets", "badnets_grid")
        train_images = os.path.join(datasets_folder, "train", POISONED_RATE, "data.npy")
        train_labels = os.path.join(datasets_folder, "train", POISONED_RATE, "labels.npy")
        test_images = os.path.join(datasets_folder, "test", POISONED_RATE, "data.npy")
        test_labels = os.path.join(datasets_folder, "test", POISONED_RATE, "labels.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    best_acc = 0  # best test accuracy
    # start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            # transforms.ToTensor(),
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
    )
    testset = TensorDataset(
        torch.tensor(cifar_10_dataset.test_images, dtype=torch.float32).permute(
            0, 3, 1, 2
        ),
        torch.tensor(cifar_10_dataset.test_labels, dtype=torch.long),
    )
    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
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
        train_loss, train_acc, training_time = train(net,epoch,trainloader)
        train_loss_data[epoch] = train_loss
        train_acc_data[epoch] = train_acc
        total_training_time += training_time

        val_loss, val_acc, testing_time = test(net,epoch,testloader)
        val_loss_data[epoch] = val_loss
        val_acc_data[epoch] = val_acc
        total_testing_time += testing_time

        scheduler.step()

    print(f"Total Training Time: {format_time(total_training_time)}")
    print(f"Total Testing Time: {format_time(total_testing_time)}")

    plot_metrics(
        train_loss_data,
        train_acc_data,
        val_loss_data,
        val_acc_data,
        total_training_time,
        total_testing_time,
    )
