#!/usr/bin/python3

from torchvision.datasets import CIFAR10
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import csv


class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset: CIFAR10,
                 y_target,
                 poisoned_rate,
                 poisoning_strategy):
        """
        Args:
            y_target
            poisoned_rate - the percantage of images we wish to transform
            poisoning_strategy - an instace of a class which can be used for transforming images (its call method must take a PIL Image type and return one)
            y_target - the class we are targeting in our backdoor attack

        """
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            download=True)
        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]
        self.y_target = y_target
        self.poisoning_strategy = poisoning_strategy

        self.total_num = len(benign_dataset)
        self.poisoned_num = int(self.total_num * poisoned_rate)
        assert self.poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(self.total_num))
        random.shuffle(tmp_list)

        # make a set of poisoned indices
        self.poisoned_indices = frozenset(tmp_list[:self.poisoned_num])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_indices:
            img = self.poisoning_strategy(img)
            target = self.y_target

        return img, target

    def save(self, filepath):
        """
        Arguments:
            filepath: String where the data should be saved (three files are created: data.npy, labels.npy and log.csv)
        """
        data_file = filepath + "/data.npy"
        target_file = filepath + "/targets.npy"
        csv_file = open(filepath + "/log.csv", 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['index', 'old label', 'new label'])

        output_images = np.empty((self.total_num, 32, 32, 3), dtype="uint8")
        output_labels = np.empty(self.total_num, dtype="uint8")

        for i in range(self.total_num):
            img, label = Image.fromarray(self.data[i]), int(self.targets[i])

            if i in self.poisoned_indices:
                img = self.poisoning_strategy(img)
                new_label = self.y_target
                self.__write_in_csv(csv_writer, i, label, new_label)

            output_images[i] = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
            output_labels[i] = new_label if i in self.poisoned_indices else label

        np.save(data_file, output_images)
        np.save(target_file, output_labels)

        return

    def __write_in_csv(self, csv_writer, index, old_label, new_label):
        csv_writer.writerow(
            [index, f"{self.classes[old_label]} ({old_label})", f"{self.classes[new_label]} ({new_label})"]
        )


if __name__ == '__main__':
    from data_poisoning.DataPoisoning import BlendCIFAR10Image

    pattern = Image.open(os.path.join('..', 'resources', 'data_poisoning', 'hello_kitty_pattern.png'))
    poisoned_train = PoisonedCIFAR10(benign_dataset=CIFAR10(root=os.path.join('..', 'datasets', 'CIFAR10'),
                                                            train=True,
                                                            download=True),
                                     y_target=1,
                                     poisoned_rate=0.2,
                                     poisoning_strategy=BlendCIFAR10Image(pattern, alpha=0.2)
                                     )

    # uncomment to show 10 sample images
    #################################
    # for _ in range(10):
    #     index = random.randint(0, poisoned_train.total_num - 1)
    #     # or random.choice(list(poisoned_train.poisoned_indices)) to see just poisoned data
    #
    #     img, target = poisoned_train[index]
    #     plt.imshow(img)
    #     plt.title(target)
    #     plt.show()
    #################################

    # uncomment to save poisoned model (warning: cpu/ram intensive!)
    # poisoned_train.save(os.path.join('..', 'datasets', 'data_poisoning_1', 'train'))

    # uncomment to test saved dataset
    #################################
    saved_imgs = np.load(os.path.join('..', 'datasets', 'data_poisoning_1', 'train_images.npy'))
    saved_targets = np.load(os.path.join('..', 'datasets', 'data_poisoning_1', 'train_targets.npy'))
    for _ in range(10):
        index = random.randint(0, len(saved_imgs))
        plt.imshow(saved_imgs[index])
        plt.title(f"original target: {saved_targets[index][0]}, new target: {saved_targets[index][1]}")
        plt.show()
    #################################
