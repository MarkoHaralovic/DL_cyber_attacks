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
from torch.utils.data import DataLoader, TensorDataset,Dataset
from torchvision import transforms, models
import numpy as np
import random
from tqdm import tqdm
import time
import copy
import csv
from datetime import datetime

from Pruning import Pruning
from FineTuningAlt import FineTuning

sys.path.append("../../../models")
from resnet18 import ResNet18


sys.path.append("../../../models_functions")
from efficient_net_functions import load_model, _train, test, evaluate_model, save_model

sys.path.append("../../../notebooks")
from Data import Data

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
    
CSV_DIR = os.path.join("..", "..", "..", config['CSV_DIR'])
CSV_PRUNING_DIR = os.path.join(CSV_DIR, config['CSV_PRUNING_DIR'])
DATASETS_DIR = os.path.join("..", "..", "..", config['DATASETS_DIR'])
CIFAR_DIR = os.path.join(DATASETS_DIR, config['CIFAR_DIR'])
POISONED_DIR = os.path.join(DATASETS_DIR, config['POISONED_DIR'])
MODEL_NAME = config['MODEL_NAME']
WEIGHT_PATH = os.path.join(config['WEIGHT_PATH'])
EXP_NAME = config['EXP_NAME']
PRUNING_RATES = config['PRUNING_RATES']
LAYER_KEYS = config['LAYER_KEYS']
TRAIN_SIZE_LIMIT = config['TRAIN_SIZE_LIMIT']
TEST_SIZE_LIMIT = config['TEST_SIZE_LIMIT']
BATCH_SIZE = config['BATCH_SIZE']
NUM_WORKERS = config['NUM_WORKERS']
EXP_NAME = config['EXP_NAME'] 
POISONED_RATE =config["POISONED_RATE"]
EPOCHS = config['FINE_TUNE_EPOCHS']
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")


class FinePruning():
   def __init__(self, train_loader, test_loader,backdoored_loader, cifar_data,model,layers_to_prune,
                original_state_dict,device="cpu"):
      self.model = model
      self.train_loader = train_loader
      self.test_loader = test_loader
      self.backdoored_loader=backdoored_loader
      self.cifar_data = cifar_data
      self.device = device
      self.model = model 
      self.layers_to_prune = layers_to_prune
      self.original_state_dict = original_state_dict

      self.criterion=nn.CrossEntropyLoss()
      self.pruning = Pruning(self.device)
      
   def evaluate_model(self,transform, data_loader_type = "train" ,asr=False):
      if data_loader_type =="train":
         data_loader = self.train_loader
      elif data_loader_type =="test":
         data_loader = self.test_loader
      elif data_loader_type =="back":
         data_loader = self.backdoored_loader
      if asr:
         acc,loss = self.pruning.evaluate_model(self.model, data_loader, device, transform)
      else:
         acc,loss = self.pruning.evaluate_model(self.model, data_loader, device, transform)
      return acc,loss
   
   def restore_model(self):
        self.model.load_state_dict(copy.deepcopy(original_state_dict))
 
   def prune(self,layer_to_prune, layer_weight_key, prune_rate):
      self.pruning.prune_layer(self.model, layer_to_prune, layer_weight_key, prune_rate,self.train_loader, device=self.device)

   def fine_tune(self, device,transformTrain,transformTest,dataloaders_dict,feature_extract = False):
        self.fineTuning = FineTuning(device = device,
                                     transformTrain = transformTrain, 
                                     transformTest = transformTest,
                                     dataloaders_dict = dataloaders_dict,
                                     feature_extract = False
                                     )
        self.fineTuning.prepare_params(self.model)
        self.model_ft, self.hist = self.fineTuning.train_model(num_epochs=EPOCHS)
        
        return self.model_ft

class IndexedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        assert len(data) == len(labels), "Data and labels must be of the same length"

    def __getitem__(self, index):
        return self.data[index], self.labels[index], index

    def __len__(self):
        return len(self.data)
    
def ASR(clean_acc, backdoor_acc):
    return float(1- backdoor_acc/clean_acc)
          
if __name__ == "__main__":
    

    # Load dataset
    print("Loading data...")
    train_images = os.path.join(CIFAR_DIR, "train","data.npy")
    train_labels = os.path.join(CIFAR_DIR, "train",  "labels.npy")
    test_images = os.path.join(CIFAR_DIR, "test",  "data.npy")
    test_labels = os.path.join(CIFAR_DIR, "test", "labels.npy")
    log_file = os.path.join(POISONED_DIR, "test", POISONED_RATE, "log.csv")

    train_images_pois = os.path.join(POISONED_DIR, "train", POISONED_RATE, "data.npy")
    train_labels_pois = os.path.join(POISONED_DIR, "train", POISONED_RATE, "labels.npy")
    test_images_pois = os.path.join(POISONED_DIR, "test", POISONED_RATE, "data.npy")
    test_labels_pois = os.path.join(POISONED_DIR, "test", POISONED_RATE, "labels.npy")
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    cifar_10_dataset = Data(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    cifar_10_dataset_pois = Data(
        train_images=train_images_pois,
        train_labels=train_labels_pois,
        test_images=test_images_pois,
        test_labels=test_labels_pois
    )
    
    transform_test = transforms.Compose(
        [
            # transforms.ToTensor(), # only needed when adding transforms directly to Data
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Train
    train_data = torch.tensor(
        cifar_10_dataset.train_images, dtype=torch.float32
    ).permute(0, 3, 1, 2)[:TRAIN_SIZE_LIMIT]
    train_labels = torch.tensor(cifar_10_dataset.train_labels, dtype=torch.long)[
        :TRAIN_SIZE_LIMIT
    ]
    # train_dataset = TensorDataset(train_data,train_labels)
    indexed_train_dataset = IndexedDataset(train_data, train_labels)
    train_loader = DataLoader(
        indexed_train_dataset,
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
    
    # Test Poisoned Dataset
    test_data_pois = torch.tensor(cifar_10_dataset_pois.test_images, dtype=torch.float32).permute(
        0, 3, 1, 2
    )[:TEST_SIZE_LIMIT]
    test_labels_pois = torch.tensor(cifar_10_dataset_pois.test_labels, dtype=torch.long)[
        :TEST_SIZE_LIMIT
    ]
    
    # test_dataset = TensorDataset(test_data,test_labels)
    indexed_test_dataset = IndexedDataset(test_data, test_labels)
    test_loader = DataLoader(
        indexed_test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
        shuffle=False
    )

    # Test Loader For Poisoned Data
    indexed_test_dataset_pois = IndexedDataset(test_data_pois, test_labels_pois)
    test_loader_pois = DataLoader(
        indexed_test_dataset_pois,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
        shuffle=False
    )
    
    # Isolated poisoned
    with open(log_file) as f:
        reader = csv.reader(f)
        header = next(iter(reader))  

        backdoored_indices = []
        backdoored_labels = []
        for row in reader:
            index = int(row[0])
            new_label = int(row[1].split()[1].strip('()'))  
            backdoored_indices.append(index)
            backdoored_labels.append(new_label)

    backdoored_data = torch.tensor(
        cifar_10_dataset.test_images[backdoored_indices], dtype=torch.float32
    ).permute(0, 3, 1, 2)[:TEST_SIZE_LIMIT]
    
    backdoored_data_labels = torch.tensor(
        cifar_10_dataset.test_labels[backdoored_indices], dtype=torch.long
    )[:TEST_SIZE_LIMIT] 

    backdoored_dataset = IndexedDataset(backdoored_data, backdoored_data_labels)
    backdoored_loader = DataLoader(
        backdoored_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
    )
    
    print(f"Len of train dataset: {len(train_data)}")
    print(f"Len of test dataset: {len(test_data)}")
    print(f"Len of backdoored dataset: {len(backdoored_data)}")

    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18()
    model.to(device)

    if device == "cuda":
        state_dict = torch.load(WEIGHT_PATH)
    else:
        state_dict = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))


    model.load_state_dict(state_dict["net"])
    print(state_dict["epoch"], state_dict["acc"])

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

    finePruning = FinePruning(
       train_loader = train_loader,
       test_loader = test_loader,
       backdoored_loader = backdoored_loader,
       cifar_data  = cifar_10_dataset,
       model = model,
       layers_to_prune = layers_to_prune,
       original_state_dict = original_state_dict,
       device="cpu")
    
    original_accuracy, original_loss = finePruning.evaluate_model(transform = transform_test, 
                                                                  data_loader_type = "test" 
                                                                  )
    print(f"Original Test Accuracy: {original_accuracy}%")
    print(f"Original Test Loss: {original_loss}%")

    print("---------------------------------------------------------------------------------------------------------------")
    
    org_backdoor_accuracy,org_backdoor_loss = finePruning.evaluate_model(transform = transform_test, 
                                                                      data_loader_type = "back"
                                                                      )
    print(f"Original Accuracy on Backdoored Data: {org_backdoor_accuracy}%")
    print(f"Original Loss on Backdoored Data: {org_backdoor_loss}%")
    
    print("---------------------------------------------------------------------------------------------------------------")
    
    org_asr = ASR(original_accuracy, org_backdoor_accuracy)
    print(f"Original ASR  : {org_asr}")

    print("---------------------------------------------------------------------------------------------------------------")
    
    print("\nStarting pruning")

    os.makedirs(CSV_PRUNING_DIR, exist_ok=True)
    csv_file_path = os.path.join(
        CSV_PRUNING_DIR, f"evaluate_pruning_{EXP_NAME}_{TIMESTAMP}.csv"
    )
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model_name",
                "pruning_rate",
                "layer_name",
                "accuracy",
                "backdoor_accuracy",
                "attack_success_rate"
            ]
        )

    for layer_idx, layer_to_prune in enumerate(layers_to_prune):
        print(f"\nPruning layer {LAYER_KEYS[layer_idx]}: ({layer_to_prune})")
        for rate in PRUNING_RATES:
            print(f"Pruning with rate {rate}")
            finePruning.prune(layer_to_prune, layers_to_prune[layer_to_prune], rate)

            print("---------------------------------------------------------------------------------------------------------------")
            print("Running on clean data")
            accuracy,loss = finePruning.evaluate_model(transform = transform_test, 
                                                        data_loader_type = "test"
                                                        )
            
            print(f"\tAccuracy {accuracy} for {LAYER_KEYS[layer_idx]} and rate {rate}")
            print(f"Test Loss: {loss} for {LAYER_KEYS[layer_idx]} and rate {rate}")

            print("---------------------------------------------------------------------------------------------------------------")
            print("Running on poisoned data")
            backdoor_accuracy,backdoor_loss = finePruning.evaluate_model(transform = transform_test, 
                                                                      data_loader_type = "back"
                                                                      )
            print("Running on poisoned data")
            print(f"\tAccuracy {backdoor_accuracy} for {LAYER_KEYS[layer_idx]} and rate {rate}")
            print(f"Test Loss: {backdoor_loss} for {LAYER_KEYS[layer_idx]} and rate {rate}")
            
            print("---------------------------------------------------------------------------------------------------------------")
            asr = ASR(accuracy, backdoor_accuracy)
            print(f"ASR  : {asr}")
            
            with open(csv_file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["model", rate, LAYER_KEYS[layer_idx], accuracy, backdoor_accuracy, asr])

            finePruning.restore_model()

    print("\nDone")
    
    pruning_accuracy, pruning_loss = finePruning.evaluate_model(
                                                                transform = transform_test, 
                                                                data_loader_type = "test" ,
                                                                )
    print(f"After Pruning  Test Accuracy on Clean Data: {pruning_accuracy}%")
    print(f"After Pruning  Test Loss on Clean Data: {pruning_loss}%")
    
    print("Running on poisoned data")
    backdoor_accuracy,backdoor_loss = finePruning.evaluate_model(transform = transform_test, 
                                                                      data_loader_type = "back"
                                                                      )
    print(f"\tAccuracy on Poisoned Data {backdoor_accuracy}")
    print(f"Test Loss on Poisoned Data: {backdoor_loss}")
            
    asr = ASR(pruning_accuracy, backdoor_accuracy)
    print(f"Final ASR after Pruning  : {asr}")

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
    
    print("Starting with Fine Tuning")
    image_datasets = {'train':indexed_train_dataset,'val':indexed_test_dataset}

    dataloaders_dict = {'train':train_loader,'val':test_loader}
    
    fineTunedModel  = FinePruning.fineTune(
        device = device,
        transformTrain = transformTrain, 
        transformTest = transformTest,
        dataloaders_dict = dataloaders_dict,
        feature_extract = False
    )
    
    print(f"Running final evaluation of fine tuned model on clean and poisoned dataset")
    ft_accuracy, ft_loss = evaluate_model(model =fineTunedModel,
                                          data_loader = test_loader, 
                                          device = device
                                          )
    print(f"Ft Test Accuracy: {ft_accuracy}%")
    print(f"Ft Test Loss: {ft_loss}%")

    ft_backdoor_accuracy,ft_backdoor_loss = evaluate_model(model =fineTunedModel,
                                                           data_loader = backdoored_loader, 
                                                           device = device
                                                          )
    print(f"Ft Accuracy on Backdoored Data: {ft_backdoor_accuracy}%")
    print(f"Ft on Backdoored Data: {ft_backdoor_loss}%")
    
    ft_asr = ASR(ft_accuracy, ft_backdoor_accuracy)
    print(f"Ft ASR  : {ft_asr}")
    
    
    