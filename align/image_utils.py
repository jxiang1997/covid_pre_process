# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import align.kernel_function

import SimpleITK as sitk
import torch as th
import torch.nn.functional as F
import numpy as np
import sys

class Image:
    """
        Class representing an image in airlab
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for an image object where two cases are distinguished:
        - Construct airlab image from an array or tensor (4 arguments)
        - Construct airlab image from an SimpleITK image (less than 4 arguments
        """
        if len(args) == 4:
            self.initializeForTensors(*args)
        elif len(args) < 4:
            self.initializeForImages(*args)


    def initializeForTensors(self, tensor_image, image_size, image_spacing, image_origin):
        """
        Constructor for torch tensors and numpy ndarrays
        Args:
        tensor_image (np.ndarray | th.Tensor): n-dimensional tensor, where the last dimensions are the image dimensions while the preceeding dimensions need to empty
        image_size (array | list | tuple): number of pixels in each space dimension
        image_spacing (array | list | tuple): pixel size for each space dimension
        image_origin (array | list | tuple): physical coordinate of the first pixel
        :return (Image): an airlab image object
        """

        # distinguish between numpy array and torch tensors
        if type(tensor_image) == np.ndarray:
            self.image = th.from_numpy(tensor_image).squeeze().unsqueeze(0).unsqueeze(0)
        elif type(tensor_image) == th.Tensor:
            self.image = tensor_image.squeeze().unsqueeze(0).unsqueeze(0)
        else:
            raise Exception("A numpy ndarray or a torch tensor was expected as argument. Got " + str(type(tensor_image)))

        self.size = image_size
        self.spacing = image_spacing
        self.origin = image_origin
        self.dtype = self.image.dtype
        self.device = self.image.device
        self.ndim = len(self.image.squeeze().shape) # take only non-empty dimensions to count space dimensions


    def initializeForImages(self, sitk_image, dtype=None, device='cpu'):
        """
        Constructor for SimpleITK image
        Note: the order of axis are flipped in order to follow the convention of numpy and torch
        sitk_image (sitk.SimpleITK.Image):  SimpleITK image
        dtype: pixel type
        device ('cpu'|'cuda'): on which device the image should be allocated
        return (Image): an airlab image object
        """
        if type(sitk_image)==sitk.SimpleITK.Image:
            self.image = th.from_numpy(sitk.GetArrayFromImage(sitk_image)).unsqueeze(0).unsqueeze(0)
            self.size = sitk_image.GetSize()
            self.spacing = sitk_image.GetSpacing()
            self.origin = sitk_image.GetOrigin()

            if not dtype is None:
                self.to(dtype, device)
            else:
                self.to(self.image.dtype, device)

            self.ndim = len(self.image.squeeze().shape)

            self._reverse_axis()
        else:
            raise Exception("A SimpleITK image was expected as argument. Got " + str(type(sitk_image)))


    def to(self, dtype=None, device='cpu'):
        """
        Converts the image tensor to a specified dtype and moves it to the specified device
        """
        if not dtype is None:
            self.image = self.image.to(dtype=dtype, device=device)
        else:
            self.image = self.image.to(device=device)
        self.dtype = self.image.dtype
        self.device = self.image.device

        return self


    def itk(self):
        """
        Returns a SimpleITK image
        Note: the order of axis is flipped back to the convention of SimpleITK
        """
        image = Image(self.image.cpu().clone(), self.size, self.spacing, self.origin)
        image._reverse_axis()
        image.image.squeeze_()

        itk_image = sitk.GetImageFromArray(image.image.numpy())
        itk_image.SetSpacing(spacing=self.spacing)
        itk_image.SetOrigin(origin=self.origin)
        return itk_image


    def numpy(self):
        """
        Returns a numpy array
        """
        return self.image.cpu().squeeze().numpy()


    @staticmethod
    def read(filename, dtype=th.float32, device='cpu'):
        """
        Static method to directly read an image through the Image class
        filename (str): filename of the image
        dtype: specific dtype for representing the tensor
        device: on which device the image has to be allocated
        return (Image): an airlab image
        """
        return Image(sitk.ReadImage(filename, sitk.sitkFloat32), dtype, device)


    def write(self, filename):
        """
        Write an image to hard drive
        Note: order of axis are flipped to have the representation of SimpleITK again
        filename (str): filename where the image is written
        """
        sitk.WriteImage(self.itk(), filename)


    def _reverse_axis(self):
        """
        Flips the order of the axis representing the space dimensions (preceeding dimensions are ignored)
        Note: the method is inplace
        """
        # reverse order of axis to follow the convention of SimpleITK
        self.image = self.image.squeeze().permute(tuple(reversed(range(self.ndim))))
        self.image = self.image.unsqueeze(0).unsqueeze(0)




"""
    Create tensor image representation
"""
def create_tensor_image_from_itk_image(itk_image, dtype=th.float32, device='cpu'):

    # transform image in a unit direction
    image_dim = itk_image.GetDimension()
    if image_dim == 2:
        itk_image.SetDirection(sitk.VectorDouble([1, 0, 0, 1]))
    else:
        itk_image.SetDirection(sitk.VectorDouble([1, 0, 0, 0, 1, 0, 0, 0, 1]))

    image_spacing = itk_image.GetSpacing()
    image_origin = itk_image.GetOrigin()

    np_image = np.squeeze(sitk.GetArrayFromImage(itk_image))
    image_size = np_image.shape

    # adjust image spacing vector size if image contains empty dimension
    if len(image_size) != image_dim:
        image_spacing = image_spacing[0:len(image_size)]

    tensor_image = th.tensor(np_image, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)


    return Image(tensor_image, image_size, image_spacing, image_origin)

"""
    Convert an image to tensor representation
"""
def read_image_as_tensor(filename, dtype=th.float32, device='cpu'):

    itk_image = sitk.ReadImage(filename, sitk.sitkFloat32)

    return create_tensor_image_from_itk_image(itk_image, dtype=dtype, device=device)