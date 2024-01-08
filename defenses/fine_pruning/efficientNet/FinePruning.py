"""
This defense is based on two types of defenses:
      - fine tuning
      - pruning
The pruning defense works as follows: the defender exercises the DNN received from the attacker with clean inputs from the validation dataset, Dvalid, 
and records the average activation of each neuron. The defender then iteratively prunes neurons from the DNN in increasing order of average activations
and records the accuracy of the pruned network in each iteration. The defense terminates when the accuracy on the validation dataset drops below a
pre-determined threshold.

Pruning has three phases :
   Phase One : the neurons in the first phase are not activated by either clean or backdoored inputs
   Phase Two : the neurons in the second phase are activated by backdoored inputs
   Phase Three : the neurons in the third phase are activated by clean inputs
    
Fine-tuning uses the pre-trained DNN weights to initialize training (instead of random initialization) and a smaller learning rate since the final weights
are expected to be relatively close to the pretrained weights. Fine-tuning is significantly faster than training a network from scratch

Fine Pruning:  first prunes the DNN returned by the attacker and then fine-tunes the pruned network.
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
import numpy as np
import random
from tqdm import tqdm
import time
import copy
import csv
from datetime import datetime

from Pruning import Pruning
from FineTuning import FineTuning

sys.path.append("../../../models")
from efficient_net_functions import load_model, _train, test, evaluate_model, save_model

sys.path.append("../../../notebooks")
from Data import Data

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
    
CSV_DIR = os.path.join(config['CSV_DIR'])
CSV_PRUNING_DIR = os.path.join(CSV_DIR, config['CSV_PRUNING_DIR'])
DATASETS_DIR = os.path.join(config['DATASETS_DIR'])
CIFAR_DIR = os.path.join(DATASETS_DIR, config['CIFAR_DIR'])
MODEL_NAME = config['MODEL_NAME']
WEIGHT_PATH = os.path.join(config['WEIGHT_PATH'])
EXP_NAME = config['EXP_NAME']
PRUNING_RATES = config['PRUNING_RATES']
LAYER_KEYS = config['LAYER_KEYS']
TRAIN_SIZE_LIMIT = config['TRAIN_SIZE_LIMIT']
TEST_SIZE_LIMIT = config['TEST_SIZE_LIMIT']
BATCH_SIZE = config['BATCH_SIZE']
NUM_WORKERS = config['NUM_WORKERS']


class FinePruning():
   def __init__(self, train_loader, test_loader, cifar_data,model,layers_to_prune,original_state_dict, device="cpu"):
      self.model = model
      self.train_loader = train_loader
      self.test_loader = test_loader
      self.cifar_data = cifar_data
      self.device = device
      self.model = model 
      self.layers_to_prune = layers_to_prune
      self.original_state_dict = original_state_dict

      # Initialize Pruning and FineTuning classes
      self.pruning = Pruning(self.device)
      self.fine_tuning = FineTuning(self.train_loader, 
                                    self.test_loader, 
                                    self.cifar_data,
                                    self.device)
   def prune(self):
        os.makedirs(CSV_PRUNING_DIR, exist_ok=True)
        csv_file_path = os.path.join(CSV_PRUNING_DIR, f"evaluate_pruning_{EXP_NAME}.csv")
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["model_name", "pruning_rate", "layer_name", "accuracy"])

        for layer_idx, layer_to_prune in enumerate(self.layers_to_prune):
            print(f"\nPruning layer {LAYER_KEYS[layer_idx]}: ({layer_to_prune})")
            for rate in PRUNING_RATES:
                print(f"Pruning with rate {rate}")
                self.pruning.prune_layer(self.model, self.train_loader, layer_to_prune, self.layers_to_prune[layer_to_prune], rate)

                loss, accuracy = evaluate_model(self.model, self.test_loader, self.device)
                print(f"\tAccuracy {accuracy}, loss {loss} for {LAYER_KEYS[layer_idx]} and rate {rate}")

                with open(csv_file_path, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["model", rate, LAYER_KEYS[layer_idx], accuracy])

                # restore model parameters
                self.model.load_state_dict(copy.deepcopy(original_state_dict))

   def fine_tune(self,learning_rate, criterion):
        self.fineTuning.build_model(model = self.model)

        optimizer = optim.Adam(self.fineTuning.model.head.classifier.parameters(), lr=learning_rate)
        fineTuning.fit_model(25, optimizer, self.train_loader, self.test_loader, criterion)

        self.fineTuning.unfreeze_model(20)

        optimizer_ft = optim.Adam(
            filter(lambda p: p.requires_grad, self.fineTuning.model.parameters()), lr=1e-5
        )

        start_time = time.time()

        self.fineTuning.fit_model(4, optimizer_ft, self.train_loader, self.test_loader, criterion)

        end_time = time.time()  
        training_time = end_time - start_time 
        print(f"Training Time for Epoch {epoch}: {training_time} seconds")
      
      
      
if __name__ == "__main__":
    """ Loading model and creating dataset """
    
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
    cifar_10_dataset.normalize()

    # Train
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

    # Load model
    print("Loading model...")

    model = load_model(
            n_classes=cifar_10_dataset.num_classes,
            model_name=MODEL_NAME,
            device=device,
            pretrained=True
            )
    model.to(device)

    state_dict = torch.load(WEIGHT_PATH)
    model.load_state_dict(state_dict)
    
    # Select layers that should be pruned
    layers_to_prune = {}
    for name, module in model.named_modules():
        if name in LAYER_KEYS:
            layers_to_prune[module] = name

    if layers_to_prune is None:
        raise AttributeError("Layers were wrongly named")

    print("Selected layers:")
    for layer_to_prune, _ in layers_to_prune.items():
        print("\tLayer to prune:", layer_to_prune)
        print("\tWeights shape:", layer_to_prune.weight.size())

    # original state of the model to go back to after pruning a layer in the following iteration
    original_state_dict = copy.deepcopy(model.state_dict())
    
    fine_pruning = FinePruning(
       train_loader = train_loader,
       test_loader = test_loader,
       cifar_data = cifar_10_dataset,
       model = model,
       layers_to_prune = layers_to_prune,
       original_state_dict = original_state_dict,
       device="cpu"
       )
    
    print("Original evaluation starting")
    # original_loss, original_accuracy = evaluate_model(model,test_loader, device)
    # print(f"Original Test Accuracy: {original_accuracy}%")
    # print(f"Original Test Loss: {original_loss}%")
    
    print("\nStarting pruning")
    fine_pruning.prune()
    print("\nPruning done")
    
    print("\nStarting fine-tuning")
    learning_rate = config['LEARNING_RATE']
    criterion = nn.CrossEntropyLoss()
    fine_pruning.fine_tune(learning_rate,criterion)
    print("\nFine-tuning done")
    

    
    

