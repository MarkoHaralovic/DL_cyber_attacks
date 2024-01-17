import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from datetime import datetime
import time
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torchvision import models

# Training
def train(net,epoch,trainloader):
    start_time = time.time()

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(
        enumerate(trainloader), desc=f"Epoch {epoch}", total=len(trainloader)
    ):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = transform_train(inputs)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(
        "Loss: %.3f | Acc: %.3f%% (%d/%d)"
        % (train_loss, 100.0 * correct / total, correct, total)
    )

    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training Time for Epoch {epoch}: {training_time} seconds")
    return train_loss, 100.0 * correct / total, training_time


def test(net,epoch,testloader):
    start_time = time.time()

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(
            enumerate(testloader), f"Validating epoch {epoch}", total=len(testloader)
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = transform_test(inputs)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(
        "Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)"
        % (test_loss, 100.0 * correct / total, correct, total)
    )

    # Save checkpoint.
    checkpoints_dir = os.path.join(os.path.curdir, "checkpoints")
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }

        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)

        torch.save(
            state,
            os.path.join(checkpoints_dir, f"resnet18_ckpt_{TIMESTAMP}.pth"),
        )
        best_acc = acc

    end_time = time.time()
    testing_time = end_time - start_time
    print(f"Testing Time for Epoch {epoch}: {testing_time} seconds\n")

    return test_loss, 100.0 * correct / total, testing_time


def plot_metrics(
    train_loss, train_acc, val_loss, val_acc, total_training_time, total_testing_time
):
    epochs = np.arange(NUM_EPOCHS)

    if max(val_loss) - min(train_loss) < 50:
        fig, (loss_ax, acc_ax) = plt.subplots(
            1, 2, figsize=(12, 5), layout="constrained"
        )
        loss_ax.plot(epochs, train_loss, label="train")
        loss_ax.plot(epochs, val_loss, label="val")
        loss_ax.set(title="Loss")

        axes = [loss_ax, acc_ax]
    else:
        # Plot train and validation loss separately
        fig, ((train_loss_ax, acc_ax), (val_loss_ax, empty)) = plt.subplots(
            2, 2, figsize=(12, 10), layout="constrained"
        )
        empty.set(visible=False)

        train_loss_ax.plot(epochs, train_loss)
        val_loss_ax.plot(epochs, val_loss)
        train_loss_ax.set(title="Train loss")
        val_loss_ax.set(title="Val loss")

        axes = [train_loss_ax, acc_ax, val_loss_ax]

    acc_ax.plot(epochs, train_acc, label="train")
    acc_ax.plot(epochs, val_acc, label="val")
    acc_ax.set_title("Accuracy")
    acc_ax.legend()

    for ax in axes:
        ax.grid(True, linestyle=":")
        ax.set_xlabel("Epoch")

    formatted_training_time = format_time(total_training_time)
    formatted_testing_time = format_time(total_testing_time)

    fig.suptitle(
        f"Resnet18 - {EXP_NAME}, {POISONED_RATE}\n[LR={LR}, WD={WD}, M={MOMENTUM}, BS={BATCH_SIZE}]",
        fontsize=14,
    )
    plt.figtext(
        0.5,
        0.01,
        f"Total Training Time: {formatted_training_time} | Total Testing Time: {formatted_testing_time}",
        ha="center",
        fontsize=10,
    )

    plt.savefig(
        os.path.join(os.path.curdir, "checkpoints", f"resnet18_metrics_{TIMESTAMP}.svg")
    )

def evaluate_model(model, data_loader, device, criterion=nn.CrossEntropyLoss()):
    """ResNet18 model evaluation

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        data_loader (DataLoader): DataLoader object providing a dataset for evaluation.
                                 The dataset should yield pairs of images and their corresponding labels.
        device (str): The device on which the model and data are loaded for evaluation.
                      Typically 'cuda' for GPU or 'cpu' for CPU.
        criterion (torch.nn.Module): The loss function used for evaluation.
                                     Default is nn.CrossEntropyLoss.

    Returns:
        tuple: A tuple containing average loss and accuracy on the provided dataset.

    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_size = 0
    resize_transform = transforms.Resize((224, 224), antialias=True)

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            # Resize images here if facing memory issues with whole dataset reshaped at once
            images = torch.stack([resize_transform(img) for img in images])
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_size += labels.size(0)

    average_loss = total_loss / total_size
    accuracy = 100 * total_correct / total_size
    return average_loss, accuracy

def load_model(
    n_classes,
    model_name="resnet18",
    device="cpu",
    pretrained=True,
    transfer_learning=False,
): 
    if pretrained:
        # Load a pre-trained ResNet model
        model = models.__dict__[model_name](pretrained=pretrained)

        if transfer_learning:
            # Change the number of output classes for transfer learning
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, n_classes)

        if torch.cuda.is_available():
            model.to(device)

    return model