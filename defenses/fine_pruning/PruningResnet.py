import copy
import csv
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
import sys

sys.path.append(
    "../../notebooks"
)  # note: run code from DL_cyber_attacks\defenses\fine_pruning, otherwise modify this line of code
from Data import Data

sys.path.append("../../models")
from resnet18 import ResNet18


CSV_DIR = os.path.join("..", "..", "csv_records")
CSV_PRUNING_DIR = os.path.join(CSV_DIR, "pruning")
DATASETS_DIR = os.path.join("..", "..", "datasets")
CIFAR_DIR = os.path.join(DATASETS_DIR, "CIFAR10", "cifar-10")

# https://ferhr-my.sharepoint.com/:u:/g/personal/ds54097_fer_hr/EYbZspxkaE9NmVQnMnuX818BPZZcWp2L613XBfoRvfeAmQ?e=ntTYzw
WEIGHT_PATH = os.path.join("..", "..", "checkpoints", "resnet18_ckpt_1221_1901.pth")

# Experiment parameters
EXP_NAME = "resnet"

PRUNING_RATES = [i / 10 for i in range(11)]
LAYER_KEYS = [
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.1.conv1",
    "layer4.1.conv2",
]

TRAIN_SIZE_LIMIT = 50000
TEST_SIZE_LIMIT = 10000
BATCH_SIZE = 128
NUM_WORKERS = 1


def evaluate_model(model, data_loader, device):
    """Efficient Net model evaluation

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        data_loader (DataLoader): DataLoader object providing a dataset for evaluation.
                                  The dataset should yield pairs of images and their corresponding labels.
        device (str): The device on which the model and data are loaded for evaluation.
                      Typically 'cuda' for GPU or 'cpu' for CPU.

    Returns:
        float: The accuracy of the model on the provided dataset, calculated as the percentage of correctly predicted samples.

    """
    model.eval()
    total = 0
    correct = 0
    i = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaulating model"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            i = i + 1

    accuracy = 100 * correct / total
    return accuracy


def prune_layer(model, layer_to_prune, layer_weight_key, prune_rate):
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
        for data, _ in tqdm(tr_loader, desc="Collecting layer outputs"):
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


if __name__ == "__main__":
    # Load dataset
    print("Loading data...")
    train_images = os.path.join(CIFAR_DIR, "train", "data.npy")
    train_labels = os.path.join(CIFAR_DIR, "train", "labels.npy")
    test_images = os.path.join(CIFAR_DIR, "test", "data.npy")
    test_labels = os.path.join(CIFAR_DIR, "test", "labels.npy")

    """
    Greške:
    (A) train_transform primjenjuje se na test dataset umjesto test_transforma
    (B) model treniran s ovim transformacijama daje dobre rezultate samo ako se iste transformacije koriste pri evaluaciji

    (A) bi se moglo rješiti td transformacije premjestimo iz Data objekta u TensorDataset objekt
    (B) se ne bi trebao događati (valjda?), ja mislin da je RandomCrop s paddingom problematičan (svakako su slike 32x32, nema ih smisla croppat) => pripazit kod sljedećeg treninga
    """
    cifar_10_dataset = Data(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        train_transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )

    # Train
    train_data = torch.tensor(
        cifar_10_dataset.train_images, dtype=torch.float32
    ).permute(0, 3, 1, 2)[:TRAIN_SIZE_LIMIT]
    train_labels = torch.tensor(cifar_10_dataset.train_labels, dtype=torch.long)[
        :TRAIN_SIZE_LIMIT
    ]

    train_dataset = TensorDataset(train_data, train_labels)
    tr_loader = DataLoader(
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18()
    model.to(device)

    state_dict = torch.load(WEIGHT_PATH)
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

    original_accuracy = evaluate_model(model, test_loader, device)
    print(f"Original Test Accuracy: {original_accuracy}%")
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
            prune_layer(model, layer_to_prune, layers_to_prune[layer_to_prune], rate)

            accuracy = evaluate_model(model, test_loader, device)
            print(f"\tAccuracy {accuracy} for {LAYER_KEYS[layer_idx]} and rate {rate}")

            with open(csv_file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["model", rate, LAYER_KEYS[layer_idx], accuracy])

            # restore model parameters
            model.load_state_dict(copy.deepcopy(original_state_dict))

    print("\nDone")
