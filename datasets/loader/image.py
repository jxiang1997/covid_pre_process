import torch
import torchvision

from PIL import Image, ImageFile


class image_loader():
    """
    image loader class
    """
    def __init__(self, args):
        self.args = args

    def get_image(self, path: str):
        """
        Returns an image item by its absolute path

        params: path - absolute path to image
        """
        image = torchvision.transforms.ToTensor()(Image.open(path).convert('RGB'))

        return image
    
    def get_images(self, paths: list):
        """
        Returns a list of images by their absolute paths

        params: paths - list of absolute paths to images
        """
        images = [self.get_image(path) for path in paths]
        # T, C, H, W
        images = torch.stack(images)

        images = images.transpose(0,1) #C, T, H, W

        return images
        