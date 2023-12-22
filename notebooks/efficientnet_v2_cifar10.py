"""
Train CIFAR10 with PyTorch using EfficientNet B0 model.

Run from DL_cyber_attacks\notebooks
"""
import numpy as np 
from tqdm import tqdm
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision 
from torchvision import transforms 
from torchvision.transforms import *
from torchvision.models import efficientnet_v2_s

from torch.utils.data import Dataset, DataLoader, random_split,TensorDataset
from torchsummary import summary

import time
import sys

from Data import Data
from auxiliary import format_time

sys.path.append("../models")
from efficient_net_functions import test,train,evaluate_model,_train,load_model

print(torch.__version__) 
print(torchvision.__version__) 

train_images = "..\\datasets\\CIFAR10\\cifar-10\\train\\data.npy"
train_labels = "..\\datasets\\CIFAR10\\cifar-10\\train\\labels.npy"
test_images = "..\\datasets\\CIFAR10\\cifar-10\\test\\data.npy"
test_labels = "..\\datasets\\CIFAR10\\cifar-10\\test\\labels.npy"
weight_path = "..\\models\\efficientnet_v2_s_cifar10.pth"
model_name = 'efficientnet_v2_s'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

cifar_10_dataset= Data(train_images=train_images,train_labels=train_labels,
                     test_images=test_images,test_labels=test_labels)


cifar_10_dataset.normalize()
cifar_10_dataset.images_to_tensor()
# cifar_10_dataset.show_images()

mean_r,mean_g, mean_b, std_r, std_g, std_b = cifar_10_dataset.mean_std(dataset="train")
print(mean_b, mean_g, mean_r, std_r, std_g, std_b)

train_data, train_labels, test_data, test_labels = cifar_10_dataset.permute_img_channels(permute_order=[0, 3, 1, 2])

test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)


model = load_model(cifar_10_dataset.num_classes,model_name,device,pretrained=True)

model.to(device)
state_dict = torch.load(weight_path)
model.load_state_dict(state_dict)
print(model)


training = False 
evaluating = True

if training:
    start_time = time.time()
    
    lr = 1e-5
    momentum = 0.9
    epochs = 1
    log_interval = 100

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975)

    for epoch in range(1, epochs + 1):
            scheduler.step(epoch)
            _train(model, epoch, optimizer, train_loader) #here function train() can also be called, look into efficient_net_functions.py to see the difference and usage
            test(model, test_loader)
    
    end_time = time.time()  
    training_time = end_time - start_time  
    formatted_training_time = format_time(training_time)
    print(f"Training time: {formatted_training_time}")
    
if evaluating:
    start_time = time.time() 
    
    accuracy = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy: {accuracy}%')
    
    end_time = time.time() 
    evaluating_time = end_time - start_time
    formatted_evaluating_time = format_time(evaluating_time)
    print(f"Evaluating time: {formatted_evaluating_time}")
