import numpy as np 
import torch
print(torch.__version__) 
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import matplotlib.pyplot as plt


dataroot = "DL_cyber_attacks\datasets\CIFAR10"
model_path = "DL_cyber_attacks\models\efficientnet_v2_s_cifar10.pth"

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

cifar10_transform = transforms.Normalize(
    mean=mean,
    std=std
)

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    cifar10_transform
])

testset = torchvision.datasets.CIFAR10(root="DL_cyber_attacks/datasets/CIFAR10/cifar-10/test/data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# Load the EfficientNet V2 Small model
model = efficientnet_v2_s(pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.summary()
model.eval()

