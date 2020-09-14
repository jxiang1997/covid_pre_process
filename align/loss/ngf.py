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

import align.transform as T
import align.utils as tu
import align.kernel_function as utils

from align.loss.pairwise_image_loss import _PairwiseImageLoss
from align.loss.factory import RegisterLoss

@RegisterLoss('ngf')
class NGF(_PairwiseImageLoss):
    r""" Implementation of the Normalized Gradient Fields image loss.
            Args:
                fixed_image (Image): Fixed image for the registration
                moving_image (Image): Moving image for the registration
                fixed_mask (Tensor): Mask for the fixed image
                moving_mask (Tensor): Mask for the moving image
                epsilon (float): Regulariser for the gradient amplitude
                size_average (bool): Average loss function
                reduce (bool): Reduce loss function to a single value
    """
    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, epsilon=1e-5,
                 size_average=True,
                 reduce=True):
        super(NGF, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask, size_average, reduce)

        self._name = "ngf"

        self._dim = fixed_image.ndim
        self._epsilon = epsilon

        if self._dim == 2:
            dx = (fixed_image.image[..., 1:, 1:] - fixed_image.image[..., :-1, 1:]) * fixed_image.spacing[0]
            dy = (fixed_image.image[..., 1:, 1:] - fixed_image.image[..., 1:, :-1]) * fixed_image.spacing[1]

            if self._epsilon is None:
                with th.no_grad():
                    self._epsilon = th.mean(th.abs(dx) + th.abs(dy))

            norm = th.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)

            self._ng_fixed_image = F.pad(th.cat((dx, dy), dim=1) / norm, (0, 1, 0, 1))

            self._ngf_loss = self._ngf_loss_2d
        else:
            dx = (fixed_image.image[..., 1:, 1:, 1:] - fixed_image.image[..., :-1, 1:, 1:]) * fixed_image.spacing[0]
            dy = (fixed_image.image[..., 1:, 1:, 1:] - fixed_image.image[..., 1:, :-1, 1:]) * fixed_image.spacing[1]
            dz = (fixed_image.image[..., 1:, 1:, 1:] - fixed_image.image[..., 1:, 1:, :-1]) * fixed_image.spacing[2]

            if self._epsilon is None:
                with th.no_grad():
                    self._epsilon = th.mean(th.abs(dx) + th.abs(dy) + th.abs(dz))

            norm = th.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + self._epsilon ** 2)

            self._ng_fixed_image = F.pad(th.cat((dx, dy, dz), dim=1) / norm, (0, 1, 0, 1, 0, 1))

            self._ngf_loss = self._ngf_loss_3d

    def _ngf_loss_2d(self, warped_image):

        dx = (warped_image[..., 1:, 1:] - warped_image[..., :-1, 1:]) * self._moving_image.spacing[0]
        dy = (warped_image[..., 1:, 1:] - warped_image[..., 1:, :-1]) * self._moving_image.spacing[1]

        norm = th.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)

        return F.pad(th.cat((dx, dy), dim=1) / norm, (0, 1, 0, 1))

    def _ngf_loss_3d(self, warped_image):

        dx = (warped_image[..., 1:, 1:, 1:] - warped_image[..., :-1, 1:, 1:]) * self._moving_image.spacing[0]
        dy = (warped_image[..., 1:, 1:, 1:] - warped_image[..., 1:, :-1, 1:]) * self._moving_image.spacing[1]
        dz = (warped_image[..., 1:, 1:, 1:] - warped_image[..., 1:, 1:, :-1]) * self._moving_image.spacing[2]

        norm = th.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + self._epsilon ** 2)

        return F.pad(th.cat((dx, dy, dz), dim=1) / norm, (0, 1, 0, 1, 0, 1))

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(NGF, self).GetCurrentMask(displacement)

        self._warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        # compute the gradient of the warped image
        ng_warped_image = self._ngf_loss(self._warped_moving_image)

        value = 0
        for dim in range(self._dim):
            value = value + ng_warped_image[:, dim, ...] * self._ng_fixed_image[:, dim, ...]

        value = 0.5 * th.masked_select(-value.pow(2), mask)

        return self.return_loss(value)