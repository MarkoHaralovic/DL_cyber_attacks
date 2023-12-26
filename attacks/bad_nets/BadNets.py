import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
from attacks.base import Base
from attacks.PoisonedCIFAR10 import PoisonedCIFAR10

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class AddCIFAR10Trigger:
    """
    Class for adding a backdoor trigger to a CIFAR10 image.

    Attributes:
        pattern: a backdoor trigger pattern, torch.Tensor of shape (C, H, W) -> (1, 32, 32)
        alpha: transparency of the trigger pattern, float32 [0, 1]

    Methods:
        __init__: initialize the backdoor trigger pattern and transparency
        __call__: add the backdoor trigger to the image

    """

    def __init__(self, pattern, alpha=1):
        self.alpha = alpha
        assert isinstance(pattern, Image.Image) or pattern is None, 'pattern should be a PIL image.'
        if pattern is None:
            self.pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
            self.pattern[0, -3:, -3:] = 255  # default pattern, 3x3 white square at the right corner
        else:
            self.pattern = F.pil_to_tensor(pattern)
            # print(f"pattern shape: {self.pattern.shape}")
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

    def __call__(self, img):
        """
        Add the backdoor trigger to the image.
            Arguments:
                img: PIL image
            Returns:
                PIL image
        """
        input_image = F.pil_to_tensor(img)
        output_image = self.add_trigger(input_image)
        return Image.fromarray(output_image.permute(1, 2, 0).numpy())

    def add_trigger(self, img):
        normalized_pattern = (self.pattern.float() / 255.0) * img.max()  # normalize the pattern
        return (img.float() + normalized_pattern).clamp(0, 255).type(torch.uint8)  # add the pattern to the image


# UTILITY FUNCTIONS
def display_images(test_image, output_image):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(test_image)
    axes[0].set_title("Original image")
    axes[1].imshow(output_image)
    axes[1].set_title("Image with the backdoor trigger")

    for ax in axes:
        ax.axis('off')

    plt.show()


def test_adding_trigger(test_image, add_square_trigger, add_grid_trigger):
    output_image = add_square_trigger(test_image)
    display_images(test_image, output_image)

    output_image = add_grid_trigger(test_image)
    display_images(test_image, output_image)


def load_CIFAR10_data(benign_root, batch_size, transform):
    trainset = torchvision.datasets.CIFAR10(root=benign_root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=benign_root, train=True, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader


if __name__ == "__main__":
    path = "../../resources/bad_nets"
    square_pattern = Image.open(f"{path}/trigger_image.png")
    grid_pattern = Image.open(f"{path}/trigger_image_grid.png")
    test_image = Image.open(f"{path}/kirby.png").convert("RGB")

    poisoned_image_class = "airplane"

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    benign_root = "../../datasets/CIFAR10/cifar-10"

    trainset, trainloader, testset, testloader = load_CIFAR10_data(benign_root, batch_size, transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    add_square_trigger = AddCIFAR10Trigger(square_pattern)
    add_grid_trigger = AddCIFAR10Trigger(grid_pattern)

    test_adding_trigger(test_image, add_square_trigger, add_grid_trigger)

    poisoned_dataset = PoisonedCIFAR10(benign_dataset=trainset,
                                       y_target=...,
                                       poisoned_rate=0.5,
                                       poisoning_strategy=add_square_trigger,
                                       poisoned_transform_index=...,
                                       poisoned_target_transform_index=...)
