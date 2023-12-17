import torch
from torchvision.transforms import functional as F
import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


class AddTrigger:
    def __init__(self):
        self.res = None
        self.weight = None

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        return (self.weight * img + self.res).type(torch.uint8)


class AddCIFAR10Trigger(AddTrigger):
    """Add watermarked trigger to CIFAR10 image.

    Args:
        pattern (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        weight (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
    """

    def __init__(self, pattern, weight):
        super(AddCIFAR10Trigger, self).__init__()

        if pattern is None:  # default pattern, 3x3 white square at the right corner
            self.pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
            self.pattern[0, -3:, -3:] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:  # if pattern is 2D, add a new dimension, a color channel
                self.pattern = self.pattern.unsqueeze(0)  # (32, 32) -> (1, 32, 32); (C, H, W)

        if weight is None:  # default weight, 3x3 white square at the right corner, use for multiplication
            self.weight = torch.zeros((1, 32, 32), dtype=torch.float32)
            self.weight[0, -3:, -3:] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:  # if weight is 2D, add a new dimension, a color channel
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


if __name__ == "__main__":
    print(os.getcwd())
    pattern = Image.open(r"../../resources/bad_nets/trigger_image.png")
    plt.imshow(pattern, interpolation="nearest")
    plt.show()
