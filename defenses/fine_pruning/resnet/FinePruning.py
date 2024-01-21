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
PRUNING_ACC_THRESHOLD = config['PRUNING_ACC_THRESHOLD']
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")


class FinePruning():
   def __init__(self, train_loader, test_loader,backdoored_loader_untargeted,backdoored_loader_targeted, cifar_data,model,layers_to_prune,
                original_state_dict,device="cpu"):
      self.model = model
      self.train_loader = train_loader
      self.test_loader = test_loader
      self.backdoored_loader_untargeted=backdoored_loader_untargeted
      self.backdoored_loader_targeted=backdoored_loader_targeted
      self.cifar_data = cifar_data
      self.device = torch.device(device)
      self.model = model 
      self.layers_to_prune = layers_to_prune
      self.original_state_dict = original_state_dict

      self.criterion=nn.CrossEntropyLoss()
      self.pruning = Pruning(self.device)
      
   def evaluate_model(self,transform, data_loader_type = "train" ,ev_ft=False):
      if data_loader_type =="train":
         data_loader = self.train_loader
      elif data_loader_type =="test":
         data_loader = self.test_loader
      elif data_loader_type =="targeted":
         data_loader = self.backdoored_loader_targeted
      elif data_loader_type =="untargeted":
         data_loader = self.backdoored_loader_untargeted
      else:
          raise Exception(f"Invalid Arguments, you sent data_loader_type : {data_loader_type}")
      if not ev_ft:
         acc,loss = self.pruning.evaluate_model(self.model, data_loader, device, transform)
      elif ev_ft:
          acc,loss = self.pruning.evaluate_model(self.model_ft, data_loader, device, transform) 
      else:
          raise Exception("Invalid Arguments")
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
    
    # Test Loader For Clean Data
    indexed_test_dataset = IndexedDataset(test_data, test_labels)
    test_loader = DataLoader(
        indexed_test_dataset,
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
        cifar_10_dataset_pois.test_images[backdoored_indices], dtype=torch.float32
    ).permute(0, 3, 1, 2)[:TEST_SIZE_LIMIT]
    
    backdoored_data_labels_untargeted = torch.tensor(
        cifar_10_dataset.test_labels[backdoored_indices], dtype=torch.long
    ) # use clean labels for evaluation

    backdoored_data_labels_targeted = torch.tensor(
        cifar_10_dataset_pois.test_labels[backdoored_indices], dtype=torch.long
    ) # use clean labels for evaluation
    
    """ Posioned Data and Clean Labels """
    backdoored_dataset_untargeted = IndexedDataset(backdoored_data, backdoored_data_labels_untargeted)
    backdoored_loader_untargeted = DataLoader(
        backdoored_dataset_untargeted,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
    )
    
    """ Posioned Data and Poisoned  Labels """
    backdoored_dataset_targeted = IndexedDataset(backdoored_data, backdoored_data_labels_targeted)
    backdoored_loader_targeted = DataLoader(
        backdoored_dataset_targeted,
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

    if device.type == "cuda":
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
       backdoored_loader_untargeted = backdoored_loader_untargeted,
       backdoored_loader_targeted = backdoored_loader_targeted,
       cifar_data  = cifar_10_dataset,
       model = model,
       layers_to_prune = layers_to_prune,
       original_state_dict = original_state_dict,
       device="cuda")
    
    original_accuracy, original_loss = finePruning.evaluate_model(transform = transform_test, 
                                                                  data_loader_type = "test" 
                                                                  )
    print(f"Original Test Accuracy: {original_accuracy}%")
    print(f"Original Test Loss: {original_loss}%")

    print("---------------------------------------------------------------------------------------------------------------")
    
    org_backdoor_accuracy_untargeted,org_backdoor_loss_untargeted = finePruning.evaluate_model(transform = transform_test, 
                                                                      data_loader_type = "untargeted"
                                                                      )
    print(f"Original Accuracy on Backdoored Data (Untargeted): {org_backdoor_accuracy_untargeted}%")
    print(f"Original Loss on Backdoored Data (Untargeted): {org_backdoor_loss_untargeted}%")
    
    print("---------------------------------------------------------------------------------------------------------------")
    
    org_asr = ASR(original_accuracy, org_backdoor_accuracy_untargeted)
    print(f"Original Untargeted ASR  : {org_asr}")

    print("---------------------------------------------------------------------------------------------------------------")
    original_backdoor_accuracy_targeted, original_backdoor_loss_targeted = finePruning.evaluate_model(transform = transform_test, 
                                                                            data_loader_type = "targeted"
                                                                            )
    print(f"Original Accuracy on Backdoored Data (Targeted): {original_backdoor_accuracy_targeted}%")
    print(f"Original Loss on Backdoored Data (Targeted) : {original_backdoor_loss_targeted}%")
    
    print(f"Original Targeted ASR  : {original_backdoor_accuracy_targeted}")

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
                "attack_success_rate_untargeted",
                "attack_success_rate_targeted"
            ]
        )

    for layer_idx, layer_to_prune in enumerate(layers_to_prune):
        print(f"\nPruning layer {LAYER_KEYS[layer_idx]}: ({layer_to_prune})")
        for rate in PRUNING_RATES:
            print(f"Pruning with rate {rate}")
            finePruning.prune(layer_to_prune, layers_to_prune[layer_to_prune], rate)

            print("---------------------------------------------------------------------------------------------------------------")
            print("Running on clean data")
            accuracy, loss = finePruning.evaluate_model(transform = transform_test, 
                                                                  data_loader_type = "test" 
                                                                  )
            print(f"\tAccuracy {accuracy} for {LAYER_KEYS[layer_idx]} and rate {rate}")
            print(f"Test Loss: {loss} for {LAYER_KEYS[layer_idx]} and rate {rate}")

            print("---------------------------------------------------------------------------------------------------------------")
            print("Running on poisoned data")
            backdoor_accuracy_untargeted,backdoor_loss_untargeted = finePruning.evaluate_model(transform = transform_test, 
                                                                      data_loader_type = "untargeted"
                                                                      )
            
            print(f"\tAccuracy {backdoor_accuracy_untargeted} for {LAYER_KEYS[layer_idx]} and rate {rate}")
            print(f"Test Loss: {backdoor_loss_untargeted} for {LAYER_KEYS[layer_idx]} and rate {rate}")
            
            print("---------------------------------------------------------------------------------------------------------------")
            asr = ASR(accuracy, backdoor_accuracy_untargeted)
            print(f"Untargeted ASR  : {asr}")
            print("---------------------------------------------------------------------------------------------------------------")
            
            backdoor_accuracy_targeted,backdoor_loss_targeted = finePruning.evaluate_model(transform = transform_test, 
                                                                      data_loader_type = "targeted"
                                                                      )
            print(f"Accuracy on Backdoored Data (Targeted): {backdoor_accuracy_targeted}%")
            print(f"Loss on Backdoored Data (Targeted) : {backdoor_loss_targeted}%")
            
            print(f"Targeted ASR  : {backdoor_accuracy_targeted}")
                    
            with open(csv_file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["model", rate, LAYER_KEYS[layer_idx], accuracy,
                                 asr,backdoor_accuracy_targeted])
                
            # Stop pruning if clean accuracy decreases by PRUNING_ACC_THRESHOLD
            if original_accuracy - accuracy > PRUNING_ACC_THRESHOLD * 100 and layer_idx == len(layers_to_prune) - 1:
                print(f"\nStopping pruning on layer {LAYER_KEYS[layer_idx]} with {PRUNING_ACC_THRESHOLD} threshold")
                break
            elif layer_idx == len(layer_to_prune) - 1 and rate == PRUNING_RATES[-1]:
                # Last pruning rate and last layer
                print(f"\nStopping pruning on layer {LAYER_KEYS[layer_idx]}. Accuracy is stil greater than the pruning threshold {PRUNING_ACC_THRESHOLD}.")
            else:
                finePruning.restore_model()

    print("\nDone")
    


    transformTrain = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transformTest = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    
    print("Starting with Fine Tuning")
    image_datasets = {'train':indexed_train_dataset,'val':indexed_test_dataset}

    dataloaders_dict = {'train':train_loader,'val':test_loader}
    
    fineTunedModel  = finePruning.fine_tune(
        device = device,
        transformTrain = transformTrain, 
        transformTest = transformTest,
        dataloaders_dict = dataloaders_dict,
        feature_extract = False
    )
    
    print(f"Running final evaluation of fine tuned model on clean and poisoned dataset")
    ft_accuracy, ft_loss = finePruning.evaluate_model(transform = transform_test, 
                                                        data_loader_type = "test",
                                                        ev_ft=True
                                                        )
    print(f"Ft Test Accuracy: {ft_accuracy}%")
    print(f"Ft Test Loss: {ft_loss}%")
    
    print("---------------------------------------------------------------------------------------------------------------")
    print("Running on poisoned data")

    ft_backdoor_accuracy_untargeted,ft_backdoor_loss_untargeted = finePruning.evaluate_model(transform = transform_test, 
                                                                      data_loader_type = "untargeted",
                                                                      ev_ft=True
                                                                      )
    print(f"Ft Untargeted Accuracy on Backdoored Data: {ft_backdoor_accuracy_untargeted}%")
    print(f"Ft Untargeted Loss on Backdoored Data: {ft_backdoor_loss_untargeted}%")
    
    print("---------------------------------------------------------------------------------------------------------------")
    
    ft_asr = ASR(ft_accuracy, ft_backdoor_accuracy_untargeted)
    print(f"Ft ASR  : {ft_asr}")
    
    print("---------------------------------------------------------------------------------------------------------------")
    
    
    ft_backdoor_accuracy_targeted,ft_backdoor_loss_targeted = finePruning.evaluate_model(transform = transform_test, 
                                                                      data_loader_type = "targeted",
                                                                      ev_ft=True
                                                                      )
    print(f"Ft Targeted Accuracy on Backdoored Data: {ft_backdoor_accuracy_targeted}%")
    print(f"Ft Targeted Loss on Backdoored Data: {ft_backdoor_loss_targeted}%")
    
    print("---------------------------------------------------------------------------------------------------------------")
    
    print(f"Ft ASR  : {ft_backdoor_accuracy_targeted}")
    
    """"Saving the model."""
    base_dir="../../../models"
    model_name="resnet18_ft"

    os.makedirs(base_dir, exist_ok=True)

    filename = f"{model_name}_{ft_accuracy:.2f}_{ft_asr:.2f}.pth"  

    save_path = os.path.join(base_dir, filename)

    torch.save(fineTunedModel.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    print("\nFinePruning results:")
    print("\tOrginal accuracy:", original_accuracy)
    print("\tFinePruned accuracy:", ft_accuracy)
    print()
    print("\tOriginal untargeted ASR:", org_asr)
    print("\tFinePruned untargeted ASR:", ft_asr)
    print()
    print("\tOriginal targeted ASR:", original_backdoor_accuracy_targeted)
    print("\tFinePruned targeted ASR:", ft_backdoor_accuracy_targeted)
    

            