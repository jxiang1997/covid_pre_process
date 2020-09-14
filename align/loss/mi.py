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
import align.utils as tu
import align.utils.kernel_function as utils

from align.loss.pairwise_image_loss import _PairwiseImageLoss
from align.loss.factory import RegisterLoss

@RegisterLoss('mi')
class MI(_PairwiseImageLoss):
    r""" Implementation of the Mutual Information image loss.
         .. math::
            \mathcal{S}_{\text{MI}} := H(F, M) - H(F|M) - H(M|F)
        Args:
            fixed_image (Image): Fixed image for the registration
            moving_image (Image): Moving image for the registration
            bins (int): Number of bins for the intensity distribution
            sigma (float): Kernel sigma for the intensity distribution approximation
            spatial_samples (float): Percentage of pixels used for the intensity distribution approximation
            background: Method to handle background pixels. None: Set background to the min value of image
                                                            "mean": Set the background to the mean value of the image
                                                            float: Set the background value to the input value
            size_average (bool): Average loss function
            reduce (bool): Reduce loss function to a single value
    """
    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, bins=64, sigma=3,
                 spatial_samples=0.1, background=None, size_average=True, reduce=True):
        super(MI, self).__init__(fixed_image, moving_image, fixed_mask, moving_mask, size_average, reduce)

        self._name = "mi"

        self._dim = fixed_image.ndim
        self._bins = bins
        self._sigma = 2*sigma**2
        self._normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
        self._normalizer_2d = 2.0 * np.pi*sigma**2

        if background is None:
            self._background_fixed = th.min(fixed_image.image)
            self._background_moving = th.min(moving_image.image)
        elif background == "mean":
            self._background_fixed = th.mean(fixed_image.image)
            self._background_moving = th.mean(moving_image.image)
        else:
            self._background_fixed = background
            self._background_moving = background

        self._max_f = th.max(fixed_image.image)
        self._max_m = th.max(moving_image.image)

        self._spatial_samples = spatial_samples

        self._bins_fixed_image = th.linspace(self._background_fixed, self._max_f, self.bins,
                                             device=fixed_image.device, dtype=fixed_image.dtype).unsqueeze(1)

        self._bins_moving_image = th.linspace(self._background_moving, self._max_m, self.bins,
                                              device=fixed_image.device, dtype=fixed_image.dtype).unsqueeze(1)

    @property
    def sigma(self):
        return self._sigma

    @property
    def bins(self):
        return self._bins

    @property
    def bins_fixed_image(self):
        return self._bins_fixed_image

    def _compute_marginal_entropy(self, values, bins):

        p = th.exp(-((values - bins).pow(2).div(self._sigma))).div(self._normalizer_1d)
        p_n = p.mean(dim=1)
        p_n = p_n/(th.sum(p_n) + 1e-10)

        return -(p_n * th.log2(p_n + 1e-10)).sum(), p

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(MI, self).GetCurrentMask(displacement)

        self._warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        moving_image_valid = th.masked_select(self._warped_moving_image, mask)
        fixed_image_valid = th.masked_select(self._fixed_image.image, mask)

        mask = (fixed_image_valid > self._background_fixed) & (moving_image_valid > self._background_moving)

        fixed_image_valid = th.masked_select(fixed_image_valid, mask)
        moving_image_valid = th.masked_select(moving_image_valid, mask)

        number_of_pixel = moving_image_valid.shape[0]

        sample = th.zeros(number_of_pixel, device=self._fixed_image.device,
                          dtype=self._fixed_image.dtype).uniform_() < self._spatial_samples

        # compute marginal entropy fixed image
        image_samples_fixed = th.masked_select(fixed_image_valid.view(-1), sample)

        ent_fixed_image, p_f = self._compute_marginal_entropy(image_samples_fixed, self._bins_fixed_image)

        # compute marginal entropy moving image
        image_samples_moving = th.masked_select(moving_image_valid.view(-1), sample)

        ent_moving_image, p_m = self._compute_marginal_entropy(image_samples_moving, self._bins_moving_image)

        # compute joint entropy
        p_joint = th.mm(p_f, p_m.transpose(0, 1)).div(self._normalizer_2d)
        p_joint = p_joint / (th.sum(p_joint) + 1e-10)

        ent_joint = -(p_joint * th.log2(p_joint + 1e-10)).sum()

        return -(ent_fixed_image + ent_moving_image - ent_joint)