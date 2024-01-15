"""
This is the implementation of fine-tuning proposed in [1].
[1] Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. RAID, 2018.

Fine-tuning uses the pre-trained DNN weights to initialize training (instead of random initialization) and a
smaller learning rate since the final weights are expected to be relatively close to the pretrained weights. 
Fine-tuning is significantly faster than training a network from scratch.

Here we use Resnet-18 for this purpose
"""

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json 
import sys

sys.path.append("../../../models")
from resnet18 import ResNet18

sys.path.append("../../../notebooks")
from Data import Data

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

DATASETS_DIR = os.path.join("..","..","..", "datasets")
CIFAR_DIR = os.path.join(DATASETS_DIR, "CIFAR10", "cifar-10")
BATCH_SIZE  = config['BATCH_SIZE_CUDA'] if torch.cuda.is_available() else config['BATCH_SIZE_CPU']
EPOCHS = config['FINE_TUNE_EPOCHS']
TRAIN_SIZE_LIMIT = config['TRAIN_SIZE_LIMIT']
TEST_SIZE_LIMIT = config['TEST_SIZE_LIMIT']
BATCH_SIZE = config['BATCH_SIZE']
NUM_WORKERS = config['NUM_WORKERS']


class FineTuning():
   def __init__(self,device = 'cpu', transformTrain = None, transformTest = None,dataloaders_dict=None, feature_extract = False):
      self.device = device
      self.transformTrain = transformTrain
      self.transformTest = transformTest
      self.dataloaders_dict = dataloaders_dict
      # Flag for feature extracting. When False, we finetune the whole model, 
      #   when True we only update the reshaped layer params
      self.feature_extract = feature_extract
      
   def train_model(self, num_epochs=25):
      since = time.time()
      self.val_acc_history = []
    
      best_model_wts = copy.deepcopy(self.model.state_dict())
      best_acc = 0.0

      for epoch in range(num_epochs):
         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
         print('-' * 10)
         
         for phase in ['train', 'val']:
               if phase == 'train':
                  self.model.train()
               else:
                  self.model.eval()

               running_loss = 0.0
               running_corrects = 0

               # Iterate over data.
               for inputs, labels ,_ in self.dataloaders_dict[phase]:
                  # labels= labels.sub(labels,1)
                  inputs = inputs.to(self.device)
                  labels = labels.to(self.device)

                  # zero the parameter gradients
                  self.optimizer.zero_grad()

                  # forward
                  # track history if only in train
                  with torch.set_grad_enabled(phase == 'train'):
                     
                     outputs = self.model(inputs)
                     loss = self.criterion(outputs, labels)

                     _, preds = torch.max(outputs, 1)

                     # backward + optimize only if in training phase
                     if phase == 'train':
                           loss.backward()
                           self.optimizer.step()

                  # statistics
                  running_loss += loss.item() * inputs.size(0)
                  running_corrects += torch.sum(preds == labels.data)

               epoch_loss = running_loss / len(self.dataloaders_dict[phase].dataset)
               epoch_acc = running_corrects.double() / len(self.dataloaders_dict[phase].dataset)

               print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

               # deep copy the model
               if phase == 'val' and epoch_acc > best_acc:
                  best_acc = epoch_acc
                  best_model_wts = copy.deepcopy(self.model.state_dict())
               if phase == 'val':
                  self.val_acc_history.append(epoch_acc)

         print()

      time_elapsed = time.time() - since
      print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
      print('Best val Acc: {:4f}'.format(best_acc))

      # load best model weights
      self.model.load_state_dict(best_model_wts)
      return self.model, self.val_acc_history
   
   def set_parameter_requires_grad(self, model, feature_extracting):
      if feature_extracting:
         for param in model.parameters():
            param.requires_grad = False
   def init_model(self, model = None):
      if model == None:
         self.model = ResNet18()
         self.model = self.model.to(self.device)
      else:
         self.model = model 
         self.model = self.model.to(self.device)
   
   def prepare_params(self,model=None):
      self.init_model(model)
      params_to_update = self.model.parameters()

      print("Params to learn:")
      if self.feature_extract:
         params_to_update = []
         for name,param in self.model.named_parameters():
            if param.requires_grad == True:
                  params_to_update.append(param)
                  print("\t",name)
                  
      else:
         for name,param in self.model.named_parameters():
            if param.requires_grad == True:
                  print("\t",name)

      self.optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
      
      self.criterion = nn.CrossEntropyLoss()
      
      
if __name__ == "__main__":
   mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
   img_size=224
   crop_size = 224
   transformTrain = transforms.Compose(
    [
     transforms.Resize(img_size),#, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
     #transforms.CenterCrop(crop_size),
     transforms.RandomRotation(20),
     transforms.RandomHorizontalFlip(0.1),
     transforms.ColorJitter(brightness=0.1,contrast = 0.1 ,saturation =0.1 ),
     transforms.RandomAdjustSharpness(sharpness_factor = 2, p = 0.1),
     transforms.ToTensor(),
     transforms.Normalize(mean,std),
     transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)
     ]
    )

   transformTest = transforms.Compose(
   [
      transforms.Resize((img_size,img_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
   ]
   )
   
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
   
   image_datasets = {'train':train_dataset,'val':test_dataset}

   dataloaders_dict = {'train':train_loader,'val':test_loader}
   
   fineTuning  = FineTuning(
      device = device,
      transformTrain = transformTrain, 
      transformTest = transformTest,
      dataloaders_dict = dataloaders_dict,
      feature_extract = False
   )
   
   fineTuning.prepare_params()
   model_ft, hist = fineTuning.train_model(num_epochs=EPOCHS)
   
   
   
   
   
   
