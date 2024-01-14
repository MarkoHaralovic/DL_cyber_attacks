import copy
import csv
import os
import sys
import torch
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from tqdm import tqdm
import json 
import torch.nn as nn

sys.path.append(
    "../../../notebooks"
)  # note: run code from DL_cyber_attacks\defenses\fine_pruning, otherwise modify this line of code
from Data import Data

sys.path.append("../../../models")
from resnet18 import ResNet18

with open('config.json', 'r') as config_file:
    config = json.load(config_file)


CSV_DIR = os.path.join("..", "..", "..", config['CSV_DIR'])
CSV_PRUNING_DIR = os.path.join(CSV_DIR, config['CSV_PRUNING_DIR'])
DATASETS_DIR = os.path.join("..", "..", "..", config['DATASETS_DIR'])
CIFAR_DIR = os.path.join(DATASETS_DIR, config['CIFAR_DIR'])
WEIGHT_PATH = config['WEIGHT_PATH']
EXP_NAME = config['EXP_NAME']
POISONED_RATE = config['POISONED_RATE']
PRUNING_RATES = config['PRUNING_RATES']
LAYER_KEYS = config['LAYER_KEYS']
TRAIN_SIZE_LIMIT = config['TRAIN_SIZE_LIMIT']
TEST_SIZE_LIMIT = config['TEST_SIZE_LIMIT']
BATCH_SIZE =1 #config['BATCH_SIZE']
NUM_WORKERS = config['NUM_WORKERS']
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")  

class Pruning():
    def __init__(self, device='cpu'):
        self.device = device
    def evaluate_model(self, model, data_loader, device, transform):
        """
        Calculate model accuracy, loss, and attacker's success rate on given dataset

        Args:
            model (torch.nn.Module): The neural network model to be evaluated.
            data_loader (DataLoader): DataLoader object providing a dataset for evaluation.
            device (str): The device on which the model and data are loaded for evaluation.
            transform: Transforms to be applied during evaluation.
            criterion: Loss function to calculate the loss.

        Returns:
            float: The accuracy of the model on the provided dataset.
            float: The average loss of the model on the provided dataset.
        """
        criterion=nn.CrossEntropyLoss()
        model.eval()
        total = 0
        correct = 0
        total_loss = 0
        total_batches = 0
        attack_successes = 0
        total_backdoored = 0

        with torch.no_grad():
            for images, labels, indices in tqdm(data_loader, desc="Evaluating model"):
                images, labels = transform(images.to(device)), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()
                total_batches += 1
                
        accuracy = 100 * correct / total
        average_loss = total_loss / total_batches

        return accuracy, average_loss


    def prune_layer(self,model, layer_to_prune, layer_weight_key, prune_rate):
        """
        Prune the specified layer of the model by setting the weights of certain channels to zero.

        Args:
            model (torch.nn.Module): The neural network model that is being pruned.

            layer_to_prune (torch.nn.Module): The specific layer within the model that is to be pruned.
                                            This is typically a convolutional layer in the last block of the model.

            layer_weight_key (str): A key that uniquely identifies the layer to prune in the model's
                                    state dictionary. This key is used to access and modify the weights
                                    of the layer directly.

            prune_rate (float): The proportion of channels to prune in the given layer. This rate should
                                be a float between 0 and 1, where 1 means pruning all channels and 0 means
                                pruning none.

        """
        with torch.no_grad():
            container = []

            def forward_hook(module, input, output):
                container.append(output)

            hook = layer_to_prune.register_forward_hook(forward_hook)

            model.eval()
            for data, _ , _ in tqdm(tr_loader, desc="Collecting layer outputs"):
                if device.type == "cuda":
                    model(data.cuda())
                else:
                    model(data)
            hook.remove()

            container = torch.cat(container, dim=0)
            activation = torch.mean(container, dim=[0, 2, 3])
            seq_sort = torch.argsort(activation)
            num_channels = len(activation)
            prunned_channels = int(num_channels * prune_rate)

            mask = torch.ones(layer_to_prune.weight.size()).to(device)
            for element in seq_sort[:prunned_channels]:
                mask[element, :, :, :] = 0

            layer_to_prune.weight.data *= mask

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
    
    loading_clean = True
    # Load dataset
    print("Loading data...")
    if loading_clean:
        train_images = os.path.join(CIFAR_DIR, "train","data.npy")
        train_labels = os.path.join(CIFAR_DIR, "train",  "labels.npy")
        test_images = os.path.join(CIFAR_DIR, "test",  "data.npy")
        test_labels = os.path.join(CIFAR_DIR, "test", "labels.npy")
        log_file = os.path.join(POISONED_DIR, "test", POISONED_RATE, "log.csv")
    else:
        train_images = os.path.join(POISONED_DIR, "train", POISONED_RATE, "data.npy")
        train_labels = os.path.join(POISONED_DIR, "train", POISONED_RATE, "labels.npy")
        test_images = os.path.join(POISONED_DIR, "test", POISONED_RATE, "data.npy")
        test_labels = os.path.join(POISONED_DIR, "test", POISONED_RATE, "labels.npy")
        log_file = os.path.join(POISONED_DIR, "test", POISONED_RATE, "log.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    cifar_10_dataset = Data(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
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

    # train_dataset = TensorDataset(train_data, train_labels)
    indexed_train_dataset = IndexedDataset(train_data, train_labels)
    tr_loader = DataLoader(
        # train_dataset,
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

    # test_dataset = TensorDataset(test_data, test_labels)
    indexed_test_dataset = IndexedDataset(test_data, test_labels)
    test_loader = DataLoader(
        # test_dataset,
        indexed_test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
    )

    # Isolated poisoned
    with open(log_file) as f:
        reader = csv.reader(f)
        header = next(iter(reader))  

        backdoored_indices = []
        backdoored_labels = []
        for row in reader:
            index = int(row[0])
            new_label = int(row[2].split()[1].strip('()'))  
            backdoored_indices.append(index)
            backdoored_labels.append(new_label)

    backdoored_data = torch.tensor(
        cifar_10_dataset.test_images[backdoored_indices], dtype=torch.float32
    ).permute(0, 3, 1, 2)[:TEST_SIZE_LIMIT]
    
    backdoored_data_labels = torch.tensor(
        cifar_10_dataset.test_labels[backdoored_indices], dtype=torch.long
    )[:TEST_SIZE_LIMIT] if not loading_clean else backdoored_labels

    # backdoored_dataset = TensorDataset(backdoored_data, backdoored_data_labels)
    backdoored_dataset = IndexedDataset(train_data, train_labels)
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

    pruning = Pruning(device)
    
    original_accuracy, original_loss= pruning.evaluate_model(model, 
                                                            test_loader,
                                                            device,
                                                            transform_test)
    print(f"Original Test Accuracy: {original_accuracy}%")
    print(f"Original Test Loss: {original_loss}%")
    
    original_backdoor_accuracy, original_backdoor_loss= pruning.evaluate_model(model, 
                                                                               backdoored_loader, 
                                                                               device,
                                                                               transform_test)
    print(f"Original Accuracy on Backdoored Data: {original_backdoor_accuracy}%")
    print(f"Original Loss on Backdoored Data: {original_backdoor_loss}%")

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
                "accuracy_clean",
                "accuracy_backdoor",
                "attack_success_rate"
            ]
        )

    for layer_idx, layer_to_prune in enumerate(layers_to_prune):
        print(f"\nPruning layer {LAYER_KEYS[layer_idx]}: ({layer_to_prune})")
        for rate in PRUNING_RATES:
            print(f"Pruning with rate {rate}")
            pruning.prune_layer(model, layer_to_prune, layers_to_prune[layer_to_prune], rate)

            accuracy,loss = pruning.evaluate_model(model,
                                                   test_loader,
                                                   device,
                                                   transform_test)
            print("Running on clean data")
            print(f"\tAccuracy {accuracy} for {LAYER_KEYS[layer_idx]} and rate {rate}")
            print(f"Test Loss: {loss} for {LAYER_KEYS[layer_idx]} and rate {rate}")

            backdoor_accuracy,backdoor_loss = pruning.evaluate_model(model, 
                                                                    backdoored_loader, 
                                                                    device,
                                                                    transform_test)
            
            print("Running on poisoned data")
            print(f"\tAccuracy {backdoor_accuracy} for {LAYER_KEYS[layer_idx]} and rate {rate}")
            print(f"Test Loss: {backdoor_loss} for {LAYER_KEYS[layer_idx]} and rate {rate}")
            
            asr = ASR(accuracy, backdoor_accuracy)
            print(f"Attack Success: {basr} for {LAYER_KEYS[layer_idx]} and rate {rate}")
            
            with open(csv_file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["model", rate, LAYER_KEYS[layer_idx], accuracy, backdoor_accuracy,asr])

            # restore model parameters
            model.load_state_dict(copy.deepcopy(original_state_dict))

    print("\nDone")
    
    pruning_accuracy, pruning_loss = pruning.evaluate_model(model, 
                                                            test_loader,
                                                            device,
                                                            transform_test,
                                                            )
    print(f"After Pruning  Test Accuracy: {pruning_accuracy}%")
    print(f"After Pruning  Test Loss: {pruning_loss}%")
