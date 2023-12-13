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

def load_model(n_classes, model_name=None, device='cpu',pretrained=True, tranfer_learning=False):
   if pretrained:
      model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', model_name, nclass=cifar_10_dataset.num_classes, skip_validation=False)
      model.to(device)
      state_dict = torch.load(weight_path)
      model.load_state_dict(state_dict)
   elif transfer_learning:
      from tensorflow.keras.applications import EfficientNetB0
      model = EfficientNetB0(include_top=False, weights='imagenet')
   else:
      from tensorflow.keras.applications import EfficientNetB0
      model = EfficientNetB0(weights='imagenet')
      