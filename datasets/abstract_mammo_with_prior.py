from abc import ABCMeta, abstractmethod
from loader.image import image_loader
from torch.utils import data
from tqdm import tqdm

import json
import numpy as np
import ipdb
import os

METAFILE_NOTFOUND_ERR = "Metadata file {} could not be parsed! Exception: {}!"

class Abstract_Mammo_Cancer_With_Prior_Dataset(data.Dataset):
    """
    Abstract dataset object for creating datasets with mammograms and prior mammograms. Dataset object is associated with the specified metadata file and also has a create_dataset method and a task. 
    On a forward pass, this dataset object will return an mammogram image concatenated with a prior of the same view. Dataset will primarily be used for image alignment task.
    """
    __metaclass__ = ABCMeta

    def __init__(self, args, split_group):
        """
            params: args - config.
            params: split_group - ['train'|'dev'|'test'].

            constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(Abstract_Mammo_Cancer_With_Prior_Dataset, self).__init__()

        args.metadata_path = os.path.join(args.metadata_dir, self.METADATA_FILENAME)
        self.split_group = split_group
        self.args = args
        self.image_loader = image_loader(args)

        try:
            self.metadata_json = json.load(open(args.metadata_path, 'r'))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.metadata_path, e))
        
        self.dataset = self.create_dataset(split_group, args.img_dir)
        if len(self.dataset) == 0:
            return
    
    @property
    @abstractmethod
    def METADATA_FILENAME(self):
        pass


    @abstractmethod
    def create_dataset(self, split_group, img_dir):
        """
        Creating the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        pass

    @staticmethod
    def set_args(args):
        """Sets any args particular to the dataset."""
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.get_image_item(index)
    
    def get_image_item(self, index):
        sample = self.dataset[index]

        if self.args.multi_image:
            x = self.image_loader.get_images(sample['paths'])
        else:
            x = self.image_loader.get_image(sample['path'])
        
        item = {'x': x}

        return item
        
    def get_image_paths_by_views(self, exam):
        """
        Get image paths of left and right CCs and MLOs

        params: exam - a dictionary with views and png file paths 

        returns: 4 lists of image paths of each view by this order: left_ccs, left_mlos, right_ccs, right_mlos. Force max 1 image per view.
        """

        def get_view(view_name):
            image_paths_w_view = [(view, image_path) for view, image_path in zip(exam['views'], exam['files']) if view.startswith(view_name)]

            image_paths_w_view = image_paths_w_view[:1]
            image_paths = [path for _ , path in image_paths_w_view]
            return image_paths

        left_ccs = get_view('L CC')
        left_mlos = get_view('L MLO')
        right_ccs = get_view('R CC')
        right_mlos = get_view('R MLO')

        return left_ccs, left_mlos, right_ccs, right_mlos




    