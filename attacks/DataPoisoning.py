#!/usr/bin/python3

import torch
from torch import nn
import torchvision
from torchvision import transforms

from PIL import Image
from attacks.bad_nets.base import Base
from PoisonedCIFAR10 import PoisonedCIFAR10

class BlendCIFAR10Image:
    """
    Blends an image together with a specified blending pattern and alpha value
    Args:
        blending_pattern (PIL_Image)
        alpha (float32 > 0 and < 1)
    """
    def __init__(self, blending_pattern, alpha):
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

class DataPoisoning(Base):
    """
    Construct poisoned datasets with blended key patterns into the targeted data.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        pattern (None | PIL.Image): Trigger pattern, PIL Image or None for default value
        alpha (None | torch.Tensor): Trigger pattern alpha, float > 0 and < 1 (default is 0.2).
        poisoned_transform_train_index (int): The position index that poisoned transform will be inserted in train dataset. Default: 0.
        poisoned_transform_test_index (int): The position index that poisoned transform will be inserted in test dataset. Default: 0.
        poisoned_target_transform_index (int): The position that poisoned target transform will be inserted. Default: 0.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,
                 y_target,
                 poisoned_rate,
                 pattern=None,
                 alpha=0.2,
                 poisoned_transform_train_index=0,
                 poisoned_transform_test_index=0,
                 poisoned_target_transform_index=0,
                 schedule=None,
                 seed=0,
                 deterministic=False):
        assert pattern is None or isinstance(pattern, Image), 'pattern should be None or 0-1 torch.Tensor.'
        assert alpha is None or isinstance(alpha, float) and alpha > 0 and alpha < 1, 'alpha should either be None or a float between 0 and 1'

        super(DataPoisoning, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        self.pattern = Image.open("../resources/data_poisoning/hello_kitty_pattern.png") if pattern is None else pattern
        self.alpha = 0.2 if alpha is None else alpha

        self.poisoned_train_dataset = PoisonedCIFAR10(
            train_dataset,
            y_target,
            poisoned_rate,
            BlendCIFAR10Image(self.pattern, self.alpha),
            poisoned_transform_train_index,
            poisoned_target_transform_index)

        self.poisoned_test_dataset = PoisonedCIFAR10(
            test_dataset,
            y_target,
            1.0,
            BlendCIFAR10Image(self.pattern, self.alpha),
            poisoned_transform_test_index,
            poisoned_target_transform_index)
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

    pattern = Image.open(r"../resources/data_poisoning/hello_kitty_pattern.png")
    poisoned_image_class = "airplane"

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    trainset_root = "../datasets/CIFAR10/cifar-10"
    testset_root = "../datasets/CIFAR10/cifar-10"

    trainset = torchvision.datasets.CIFAR10(root=trainset_root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=testset_root, train=True, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    data_poisoning = DataPoisoning(
        train_dataset=trainset,
        test_dataset=testset,
        model=None,
        loss=nn.CrossEntropyLoss(),
        y_target=1,  # all poisoned images will be labeled as "airplane"
        poisoned_rate=0.05,
        pattern=pattern,
        alpha=0.3,
        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        poisoned_target_transform_index=0,
        schedule=None,
        seed=666
    )

    poisoned_train_dataset, poisoned_test_dataset = data_poisoning.get_poisoned_dataset()
    # TODO: save poisoned dataset as .npy files
    # TODO: add code to specify epochs/iterations and train the model
