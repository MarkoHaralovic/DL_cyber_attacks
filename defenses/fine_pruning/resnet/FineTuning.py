import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import sys
import json 

sys.path.append("../../../models")
from resnet18 import ResNet18
sys.path.append("../../../notebooks")
from Data import Data

with open('config.json', 'r') as config_file:
    config = json.load(config_file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS_DIR = os.path.join("..","..","..", "datasets")
CIFAR_DIR = os.path.join(DATASETS_DIR, "CIFAR10", "cifar-10")
TRAIN_SIZE_LIMIT = config['TRAIN_SIZE_LIMIT']
TEST_SIZE_LIMIT = config['TEST_SIZE_LIMIT']
BATCH_SIZE = config['BATCH_SIZE']
NUM_WORKERS = config['NUM_WORKERS']
BATCH_SIZE_TRAIN = 128 if torch.cuda.is_available() else 64
BATCH_SIZE_TEST = 64 if torch.cuda.is_available() else 32


class FineTuning:
    def __init__(self, model, train_loader, test_loader, device="cpu"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def build_model(self):
        # Freeze all the layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer
        num_ftrs = self.model.linear.in_features
        self.model.linear = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes
        self.model = self.model.to(self.device)

    def fit_model(self, epochs, optimizer, criterion):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
            print('Epoch {}/{} - Loss: {:.4f} Acc: {:.4f}'.format(epoch, epochs - 1, epoch_loss, epoch_acc))

            # Validation phase
            self.model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)

            epoch_loss = running_loss / len(self.test_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.test_loader.dataset)
            print('Validation - Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        print('Training complete')

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet requires images of size 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    print("Loading data...")
    train_images = os.path.join(CIFAR_DIR, "train", "data.npy")
    train_labels = os.path.join(CIFAR_DIR, "train", "labels.npy")
    test_images = os.path.join(CIFAR_DIR, "test", "data.npy")
    test_labels = os.path.join(CIFAR_DIR, "test", "labels.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    cifar_10_dataset = Data(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    train_data = torch.tensor(
        cifar_10_dataset.train_images, dtype=torch.float32
    ).permute(0, 3, 1, 2)[:TRAIN_SIZE_LIMIT]
    train_labels = torch.tensor(cifar_10_dataset.train_labels, dtype=torch.long)[
        :TRAIN_SIZE_LIMIT
    ]

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
    )

    # Test
    test_data = torch.tensor(cifar_10_dataset.test_images, dtype=torch.float32).permute(
        0, 3, 1, 2
    )[:TEST_SIZE_LIMIT]
    test_labels = torch.tensor(cifar_10_dataset.test_labels, dtype=torch.long)[
        :TEST_SIZE_LIMIT
    ]

    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
    )

    
    net = ResNet18()
    net = net.to(device)
    
    # Initialize FineTuning
    fine_tuning = FineTuning(
       model = net,
       train_loader = train_loader,
       test_loader =  test_loader, 
       device = device
       )

    fine_tuning.build_model()

    optimizer = optim.SGD(fine_tuning.model.linear.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    fine_tuning.fit_model(10, optimizer, criterion)  # Train for 10 epochs
