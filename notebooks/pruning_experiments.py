import copy
import csv
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from Data import Data


CSV_DIR = os.path.join("..", "csv_records")
CSV_PRUNING_DIR = os.path.join(CSV_DIR, "pruning")
DATASETS_DIR = os.path.join("..", "datasets")
CIFAR_DIR = os.path.join(DATASETS_DIR, "CIFAR10", "cifar-10")

MODEL_NAME = "efficientnet_v2_s"
WEIGHT_PATH = os.path.join("..", "models", "efficientnet_v2_s_cifar10.pth")

# Experiment parameters
EXP_NAME = "initial_limit_128"

PRUNING_RATES = [i / 10 for i in range(11)]
LAYER_KEYS = [
    "blocks.39.block.depth_wise.0",
    # "blocks.39.block.linear_bottleneck.0",
    # "blocks.39.block.se.fc1",  # ova dva potupno povezana sloja  mola bi biti zeznuta jer se nadovezuju jedan na drugog
    # "blocks.39.block.se.fc2",
    "blocks.39.block.point_wise.0",
]

TRAIN_SIZE_LIMIT = 128
TEST_SIZE_LIMIT = 128
BATCH_SIZE = 32
NUM_WORKERS = 1


def evaluate_model(model, data_loader, device):
    model.eval()
    total = 0
    correct = 0
    i = 0
    resize_transform = transforms.Resize((224, 224), antialias=True)

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            # Resize images here if facing memory issues with whole dataset reshaped at once
            images = torch.stack([resize_transform(img) for img in images])
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            i = i + 1

    accuracy = 100 * correct / total
    return accuracy


def prune_layer(model, layer_to_prune, prune_rate):
    with torch.no_grad():
        container = []

        def forward_hook(module, input, output):
            container.append(output)

        hook = layer_to_prune.register_forward_hook(forward_hook)

        model.eval()
        for data, _ in tr_loader:
            if device == "cuda":
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
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load(
        "hankyul2/EfficientNetV2-pytorch",
        MODEL_NAME,
        nclass=cifar_10_dataset.num_classes,
        skip_validation=False,
    )
    model.to(device)

    state_dict = torch.load(WEIGHT_PATH)
    model.load_state_dict(state_dict)

    # Select layers that should be pruned
    layers_to_prune = []
    for name, module in model.named_modules():
        if name in LAYER_KEYS:
            layers_to_prune.append(module)

    if layers_to_prune is None:
        raise AttributeError("Layers were wrongly named")

    print("Selected layers:")
    for layer_to_prune in layers_to_prune:
        print("\tLayer to prune:", layer_to_prune)
        print("\tWeights shape:", layer_to_prune.weight.size())
    print()

    # original state of the model to go back to after pruning a layer in the following iteration
    original_state_dict = copy.deepcopy(model.state_dict())

    original_accuracy = evaluate_model(model, test_loader, device)
    print(f"Original Test Accuracy: {original_accuracy}%")
    print("\nStarting pruning")

    os.makedirs(CSV_PRUNING_DIR, exist_ok=True)
    csv_file_path = os.path.join(
        CSV_PRUNING_DIR, f"evaluate_pruning_{EXP_NAME}.csv"
    )  # f"../csv_records/pruning/evaluate_pruning_{EXP_NAME}.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model_name", "pruning_rate", "layer_name", "accuracy"])

        for layer_idx, layer_to_prune in enumerate(layers_to_prune):
            print(f"Pruning layer {LAYER_KEYS[layer_idx]}: ({layer_to_prune})")
            for rate in tqdm(PRUNING_RATES):
                prune_layer(model, layer_to_prune, rate)
                accuracy = evaluate_model(model, test_loader, device)

                writer.writerow(["model", rate, LAYER_KEYS[layer_idx], accuracy])

                # restore model parameters
                model.load_state_dict(copy.deepcopy(original_state_dict))

            print()

    print("Done")
