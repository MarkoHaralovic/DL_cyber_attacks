import copy
import csv
import os
import json
import sys
import torch
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

sys.path.append("../../../models")
from efficient_net_functions import load_model, _train, test, evaluate_model, save_model

sys.path.append(
    "../../../notebooks"
) # note: run code from DL_cyber_attacks\defenses\fine_pruning, otherwise modify this line of code
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

class Pruning():
    def __init__(self,device='cpu'):
        self.device=device

    def prune_layer(self, model,train_loader,layer_to_prune, layer_weight_key, prune_rate):
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
            for data, _ in tqdm(train_loader, desc="Collecting layer outputs"):
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
            if layer_weight_key == "blocks.39.block.depth_wise.0":
                model.blocks[39].block.depth_wise[0].weight.data *= mask
            elif layer_weight_key == "blocks.39.block.point_wise.0":
                model.blocks[39].block.point_wise[0].weight.data *= mask
            elif layer_weight_key == "blocks.39.block.linear_bottleneck.0":
                model.blocks[39].block.linear_bottleneck[0].weight.data *= mask


if __name__ == "__main__":
    # Load dataset
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
    
    model =  load_model(
            n_classes=cifar_10_dataset.num_classes,
            model_name=MODEL_NAME,
            device=device,
            pretrained=True
            )
    model.to(device)

    state_dict = torch.load(WEIGHT_PATH)
    model.load_state_dict(state_dict)

    #initialize Pruning class
    pruning = Pruning(device)

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

    original_loss, original_accuracy = evaluate_model(model, test_loader, device)
    print(f"Original Test Accuracy: {original_accuracy}%")
    print(f"Original Test Loss: {original_loss}%")
    print("\nStarting pruning")

    os.makedirs(CSV_PRUNING_DIR, exist_ok=True)
    csv_file_path = os.path.join(CSV_PRUNING_DIR, f"evaluate_pruning_{EXP_NAME}.csv")
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model_name", "pruning_rate", "layer_name", "accuracy"])

    for layer_idx, layer_to_prune in enumerate(layers_to_prune):
        print(f"\nPruning layer {LAYER_KEYS[layer_idx]}: ({layer_to_prune})")
        for rate in PRUNING_RATES:
            print(f"Pruning with rate {rate}")
            pruning.prune_layer(model, train_loader, layer_to_prune, layers_to_prune[layer_to_prune], rate)

            loss, accuracy = evaluate_model(model, test_loader, device)
            print(f"\tAccuracy {accuracy}, loss {loss} for {LAYER_KEYS[layer_idx]} and rate {rate}")

            with open(csv_file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["model", rate, LAYER_KEYS[layer_idx], accuracy])

            # restore model parameters
            model.load_state_dict(copy.deepcopy(original_state_dict))

    print("\nDone")
