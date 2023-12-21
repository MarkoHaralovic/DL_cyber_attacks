#!/usr/bin/python3

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from PIL import Image
import random
import copy
import numpy as np

class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 poisoning_strategy,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        """
        Args:
            benign_dataset - the dataset we wish to poison (CIFAR10 is the intended dataset)
            y_target
            poisoned_rate - the percantage of images we wish to transform
            poisoning_strategy - an instace of a class which can be used for transforming images (its call method must take a PIL Image type and return one)
            poisoned_transform_index - the class index we are targeting
            poisoned_target_transform_index - the class index we are transforming the data to

        """
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        self.total_num = len(benign_dataset)
        self.poisoned_num = int(self.total_num * poisoned_rate)
        assert self.poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(self.total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:self.poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, poisoning_strategy)

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, lambda: y_target)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

    def save(self, filepath):
        out = []

        for index in range(self.total_num):
            curr_img, target = self.data[index], int(self.targets[index])

            curr_img = Image.fromarray(curr_img)
            if index in self.poisoned_set:
                curr_img = self.poisoned_transform(curr_img)
                target = self.poisoned_target_transform(target)
            else:
                pass

            img_np = np.array(curr_img.getdata()).reshape(curr_img.size[0], curr_img.size[1], 3)
            out.append(img_np)

        np.save(filepath, out)
        return
