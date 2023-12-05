"""
         Some methods to grasp better sense of the data and data features.
"""

import numpy as np
from PIL import Image
from Data import Data



train_images = "DL_cyber_attacks\\datasets\\CIFAR10\\cifar-10\\train\\data.npy"
train_labels = "DL_cyber_attacks\\datasets\\CIFAR10\\cifar-10\\train\\labels.npy"
test_images = "DL_cyber_attacks\\datasets\\CIFAR10\\cifar-10\\test\\data.npy"
test_labels = "DL_cyber_attacks\\datasets\\CIFAR10\\cifar-10\\test\\labels.npy"


cifar_10_dataset= Data(train_images=train_images,train_labels=train_labels,
                     test_images=test_images,test_labels=test_labels)

# Vizualize one image per class present in the dataset
# cifar_10_dataset.show_images()

print(f"Number of classes in CIFAR10 dataset: {cifar_10_dataset.num_classes}")
print("Classes in CIFAR10 dataset : ", cifar_10_dataset.classes)

print(f"Shape of images in CIFAR10 dataset: {cifar_10_dataset.shape}")

# Vizualization of n images of a class

print("Vizualize n images of any class, I chose 10 instances of frogs in this case")

cifar_10_dataset.visualize_class_images(class_name="frog", n=10)
