#!/usr/bin/python3

import torch
import torchvision
from torchvision import transforms

import os
from PIL import Image
from PoisonedCIFAR10 import PoisonedCIFAR10

class BlendCIFAR10Image:
    """
    Blends an image together with a specified blending pattern and alpha value
    Args:
        blending_pattern (PIL_Image)
        alpha (float32 > 0 and < 1)
    """
    def __init__(self, blending_pattern, alpha=0.2):
        if blending_pattern is None or alpha <= 0 or alpha >= 1:
            raise RuntimeError

        self.blending_pattern = blending_pattern
        self.alpha = alpha
        return

    def __call__(self, img):
        """
        Blend the image with the blending pattern with which the instance was initialized.
        Args:
            img (PIL.Image)
        Return:
            new_img (PIL.Image)
        """
        temp_resized = self.blending_pattern.resize(img.size)
        new_img = Image.blend(img, temp_resized, self.alpha)
        return new_img

if __name__ == '__main__':
    # uncomment in order to test image blending
    ############################################
    # blending = Image.open("../../resources/data_poisoning/hello_kitty_pattern.png") # TODO: add different patterns for testing
    #
    # done = False
    # while(not done):
    #     test_path = input("Write path to test image: ")
    #     try:
    #         test_img = Image.open(test_path)
    #         done = True
    #     except:
    #         print("Please enter a valid path")
    #
    # blender = BlendCIFAR10Image(blending, 0.2)
    # output_image = blender(test_img)
    #
    # print("Output image should show up now...")
    # output_image.show()
    #############################################

    pattern = Image.open(os.path.join('..', 'resources', 'data_poisoning', 'hello_kitty_pattern'))
    poisoned_image_class = "airplane"

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    benign_root = os.path.join('..', 'datasets', 'CIFAR10', 'cifar-10')

    trainset = torchvision.datasets.CIFAR10(root=benign_root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=benign_root, train=True, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    poisoned_trainset = PoisonedCIFAR10(benign_dataset=trainset,
                                        y_target=1, # airplane
                                        poisoned_rate=0.2,
                                        poisoning_strategy=BlendCIFAR10Image(pattern),
                                        )

    # TODO: save poisoned dataset as .npy files
    # TODO: add code to specify epochs/iterations and train the model
