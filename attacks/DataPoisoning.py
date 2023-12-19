from PIL import Image

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

if __name__ == '__main__':
    blending = Image.open("../../resources/data_poisoning/hello_kitty_pattern.png") # TODO: add different patterns for testing

    done = False
    while(not done):
        test_path = input("Write path to test image: ")
        try:
            test_img = Image.open(test_path)
            done = True
        except:
            print("Please enter a valid path")

    blender = BlendCIFAR10Image(blending, 0.2)
    output_image = blender(test_img)

    print("Output image should show up now...")
    output_image.show()
