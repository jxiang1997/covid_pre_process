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

import torch as th
import torch.nn.functional as F

import numpy as np

import align.transform.transform as T
import align.transform.utils as tu
import align.utils.kernel_function as utils

from align.loss.pairwise_image_loss import _PairwiseImageLoss
from align.loss.factory import RegisterLoss

@RegisterLoss('ssim')
class SSIM(_PairwiseImageLoss):
    r""" Implementation of the Structual Similarity Image Measure loss.
        Args:
                fixed_image (Image): Fixed image for the registration
                moving_image (Image): Moving image for the registration
                fixed_mask (Tensor): Mask for the fixed image
                moving_mask (Tensor): Mask for the moving image
                sigma (float): Sigma for the kernel
                kernel_type (string): Type of kernel i.e. gaussian, box
                alpha (float): Controls the influence of the luminance value
                beta (float): Controls the influence of the contrast value
                gamma (float): Controls the influence of the structure value
                c1 (float): Numerical constant for the luminance value
                c2 (float): Numerical constant for the contrast value
                c3 (float): Numerical constant for the structure value
                size_average (bool): Average loss function
                reduce (bool): Reduce loss function to a single value
    """
    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None,
                 sigma=[3], dim=2, kernel_type="box", alpha=1, beta=1, gamma=1, c1=0.00001, c2=0.00001,
                 c3=0.00001, size_average=True, reduce=True, ):
        super(SSIM, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask, size_average, reduce)

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        self._c1 = c1
        self._c2 = c2
        self._c3 = c3

        self._name = "sim"
        self._kernel = None

        dim = dim
        sigma = np.array(sigma)

        if sigma.size != dim:
            sigma_app = sigma[-1]
            while sigma.size != dim:
                sigma = np.append(sigma, sigma_app)

        if kernel_type == "box":
            kernel_size = sigma * 2 + 1
            self._kernel = th.ones(*kernel_size.tolist()) \
                           / float(np.product(kernel_size) ** 2)
        elif kernel_type == "gaussian":
            self._kernel = utils.gaussian_kernel(sigma, dim, asTensor=True)

        self._kernel.unsqueeze_(0).unsqueeze_(0)

        self._kernel = self._kernel.to(dtype=self._dtype, device=self._device)

        # calculate mean and variance of the fixed image
        self._mean_fixed_image = F.conv2d(self._fixed_image.image, self._kernel)
        self._variance_fixed_image = F.conv2d(self._fixed_image.image.pow(2), self._kernel) \
                                     - (self._mean_fixed_image.pow(2))

    def forward(self, displacement):
        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(SSIM, self).GetCurrentMask(displacement)
        mask = ~mask
        mask = mask.to(dtype=self._dtype, device=self._device)

        self._warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        mask = F.conv2d(mask, self._kernel)
        mask = mask == 0

        mean_moving_image = F.conv2d(self._warped_moving_image, self._kernel)

        variance_moving_image = F.conv2d(self._warped_moving_image.pow(2), self._kernel) - (
            mean_moving_image.pow(2))

        mean_fixed_moving_image = F.conv2d(self._fixed_image.image * self._warped_moving_image, self._kernel)

        covariance_fixed_moving = (mean_fixed_moving_image - mean_moving_image * self._mean_fixed_image)

        luminance = (2 * self._mean_fixed_image * mean_moving_image + self._c1) / \
                    (self._mean_fixed_image.pow(2) + mean_moving_image.pow(2) + self._c1)

        contrast = (2 * th.sqrt(self._variance_fixed_image + 1e-10) * th.sqrt(
            variance_moving_image + 1e-10) + self._c2) / \
                   (self._variance_fixed_image + variance_moving_image + self._c2)

        structure = (covariance_fixed_moving + self._c3) / \
                    (th.sqrt(self._variance_fixed_image + 1e-10) * th.sqrt(
                        variance_moving_image + 1e-10) + self._c3)

        sim = luminance.pow(self._alpha) * contrast.pow(self._beta) * structure.pow(self._gamma)

        value = -1.0 * th.masked_select(sim, mask)

        return self.return_loss(value)