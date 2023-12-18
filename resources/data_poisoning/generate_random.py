#!/usr/bin/python3

from PIL import Image
import random

def generate_random_pattern(width, height):
    """
    A utility function which takes a dimension for an image and returns a random pattern used for data poisoning.
    Args
        width - a non-negative int32
        height - a non-negative int32
    """
    # Create a new image with a black background
    img = Image.new('RGB', (width, height), 'black')
    pixels = img.load()

    # Generate a random color for each pixel
    for x in range(width):
        for y in range(height):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            pixels[x, y] = (r, g, b)

    return img

def save_image(img, filename):
    img.save(filename)

if __name__ == "__main__":
    random_pattern = generate_random_pattern(32, 32)

    random_pattern.show()
    print("Showing generated image now. Would you like to save it to a file? (y/N)", end='')
    response = input()

    if response.lower() == "y":
        save_image(random_pattern, "random_pattern.png")
