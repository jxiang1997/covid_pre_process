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
        image = Image.open(path)

        return image
    
    def get_images(self, paths: list):
        """
        Returns a list of images by their absolute paths

        params: paths - list of absolute paths to images
        """
        images = [self.get_image(path) for path in paths]
        image = torch.stack(images)

        if self.args.num_images > 1:
            channels_times_images, _, H, W = images.size()
            assert channels_times_images == self.args.num_chan * self.args.num_images
            images = images.view([self.args.num_images, self.args.num_chan, H, W]) # T, C, H, W
            images = images.transpose(0,1) #C, T, H, W

        return images
        