"""
This is the implement of fine-tuning proposed in [1].
[1] Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. RAID, 2018.

Fine-tuning uses the pre-trained DNN weights to initialize training (instead of random initialization) and a
smaller learning rate since the final weights are expected to be relatively close to the pretrained weights. 
Fine-tuning is significantly faster than training a network from scratch.

We use Efficient Net B0 for this purpose
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
import numpy as np
import random
from tqdm import tqdm
import sys

sys.path.append("../../models")
from efficient_net_functions import load_model, _train, test, evaluate_model, save_model

sys.path.append("../../notebooks")
from Data import Data


class FineTuning:
    """
    FineTuning process for a neural network model using PyTorch.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
            This should provide batches of data in the form of (images, labels) during training.

        test_loader (DataLoader): DataLoader for the testing dataset.
            Similar to train_loader, it provides batches of data for evaluating the model.

        cifar_data: Object containing the CIFAR-10 dataset.
            This is expected to have properties or methods to access the training and testing data,
            as well as any required preprocessing or transformations.

        device (str): The computing device to use for training and evaluation.
            Typical values are 'cpu' or 'cuda' (for using NVIDIA GPU).
            Default is 'cpu'.

    Attributes:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        cifar_data: Dataset object for CIFAR-10.
        model (torch.nn.Module): The neural network model for fine-tuning.
        device (str): Device type ('cpu' or 'cuda').
    """

    def __init__(self, train_loader, test_loader, cifar_data, device="cpu"):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cifar_data = cifar_data
        self.device = device

        self.train_loss, self.valid_loss = [], []
        self.train_acc, self.valid_acc = [], []

    def build_model(self):
        self.model = load_model(
            n_classes=self.cifar_data.num_classes,
            model_name="efficientnet_v2_s",
            device=self.device,
            pretrained=True,
            transfer_learning=True,
        )
        # print(self.model)
        for param in self.model.parameters():
            param.requires_grad = False  # True ???

        # Replace the classifier
        num_features = self.model.head.classifier.in_features
        self.model.head.classifier = nn.Linear(
            num_features, self.cifar_data.num_classes
        ).to(self.device)
        # self.model.classifier[1] = nn.Linear(in_features=num_features, out_features=self.cifar_data.num_classes).to(self.device)
        self.model = self.model.to(self.device)

        self.count_parameters()

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"{total_trainable_params:,} training parameters.")

    def fit_model(self, epochs, optimizer, train_loader, val_loader, criterion):
        for epoch in range(epochs):
            epoch_loss, epoch_acc = _train(
                self.model, epoch, optimizer, train_loader, criterion
            )
            self.train_loss.append(epoch_loss)
            self.train_acc.append(epoch_acc)
            print(
                f"Epoch : {epoch}, Epoch loss : {epoch_loss}, Epoch accuracy : {epoch_acc}"
            )
            val_loss, val_accuracy = evaluate_model(
                self.model, val_loader, self.device, criterion
            )
            self.valid_loss.append(val_loss)
            self.valid_acc.append(val_accuracy)
            print(f"Epoch {epoch}: Validation Accuracy: {val_accuracy}%")
            print("-" * 50)
        save_model(epochs, self.model, optimizer, criterion)

    def unfreeze_model(self, unfreeze_layers=20):
        # Unfreeze the last 20 layers
        for param in list(self.model.parameters())[-unfreeze_layers:]:
            param.requires_grad = True


train_images = "..\\..\\datasets\\CIFAR10\\cifar-10\\train\\data.npy"
train_labels = "..\\..\\datasets\\CIFAR10\\cifar-10\\train\\labels.npy"
test_images = "..\\..\\datasets\\CIFAR10\\cifar-10\\test\\data.npy"
test_labels = "..\\..\\datasets\\CIFAR10\\cifar-10\\test\\labels.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

cifar_10_dataset = Data(
    train_images=train_images,
    train_labels=train_labels,
    test_images=test_images,
    test_labels=test_labels,
)


cifar_10_dataset.normalize()

train_data, train_labels, test_data, test_labels = cifar_10_dataset.to_tensor_permute(permute=True, permute_order=[0, 3, 1, 2])

test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

learning_rate = 1e-2  # treba biti puno manji LR
criterion = nn.CrossEntropyLoss()

fineTuning = FineTuning(train_loader, test_loader, cifar_10_dataset, device)
fineTuning.build_model()

optimizer = optim.Adam(fineTuning.model.head.classifier.parameters(), lr=learning_rate)
fineTuning.fit_model(25, optimizer, train_loader, test_loader, criterion)

fineTuning.unfreeze_model(20)

optimizer_ft = optim.Adam(
    filter(lambda p: p.requires_grad, fineTuning.model.parameters()), lr=1e-5
)
fineTuning.fit_model(4, optimizer_ft, train_loader, test_loader, criterion)
