import numpy as np 
import torch
print(torch.__version__) 
from tqdm import tqdm
import PIL.Image as Image
import torchvision 
from torchvision import transforms 
from torchvision.transforms import *
from torchvision.models import efficientnet_v2_s
from torch.utils.data import Dataset, DataLoader, random_split,TensorDataset
from torchsummary import summary
from Data import Data
import torch.nn as nn
import torch.optim as optim

print(torch.__version__) 
print(torchvision.__version__) 



train_images = "DL_cyber_attacks\\datasets\\CIFAR10\\cifar-10\\train\\data.npy"
train_labels = "DL_cyber_attacks\\datasets\\CIFAR10\\cifar-10\\train\\labels.npy"
test_images = "DL_cyber_attacks\\datasets\\CIFAR10\\cifar-10\\test\\data.npy"
test_labels = "DL_cyber_attacks\\datasets\\CIFAR10\\cifar-10\\test\\labels.npy"
weight_path = "DL_cyber_attacks\\models\\efficientnet_v2_s_cifar10.pth"
model_name = 'efficientnet_v2_s'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# didn't use this transformations since Data class has these implemented, but this can be used as well

train_transform = Compose([
    Resize((224, 224), antialias=True),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])
test_transform = Compose([
    Resize((224, 224), antialias=True), 
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cifar_10_dataset= Data(train_images=train_images,train_labels=train_labels,
                     test_images=test_images,test_labels=test_labels)


cifar_10_dataset.normalize()
# cifar_10_dataset.show_images()

mean_r,mean_g, mean_b, std_r, std_g, std_b = cifar_10_dataset.mean_std(dataset="train")
print(mean_b, mean_g, mean_r, std_r, std_g, std_b)

# Creating DatLoaders for trainig and test data 
test_data = torch.tensor(cifar_10_dataset.test_images, dtype=torch.float32).permute(0, 3, 1, 2)
test_labels = torch.tensor(cifar_10_dataset.test_labels, dtype=torch.long)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_data = torch.tensor(cifar_10_dataset.train_images, dtype=torch.float32).permute(0, 3, 1, 2)
train_labels = torch.tensor(cifar_10_dataset.train_labels, dtype=torch.long)
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)


model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', model_name, nclass=cifar_10_dataset.num_classes, skip_validation=False)
model.to(device)
state_dict = torch.load(weight_path)
model.load_state_dict(state_dict)
print(model)

def train(model, epoch, optimizer, train_loader, criterion=nn.CrossEntropyLoss()):
    total_loss = 0
    total_size = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))
            
def test(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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


# lr = 1e-5
# momentum = 0.9
# epochs = 1
# log_interval = 100

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975)

# for epoch in range(1, epochs + 1):
#         scheduler.step(epoch)
#         train(model, epoch, optimizer, train_loader)
#         test(model, test_loader)
        
accuracy = evaluate_model(model, test_loader, device)
print(f'Test Accuracy: {accuracy}%')
